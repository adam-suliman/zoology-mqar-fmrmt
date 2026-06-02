"""Replay baselines for formal permuted class-incremental AR.

At continual stage t, the train loader contains current-stage examples plus a
replay buffer. Evaluation remains the standard stage-local matrix used by the
formal permuted AR runs.

Three replay source modes are supported:

- ``oracle``: old-stage replay examples are freshly generated for each later
  stage from the synthetic task generator, balanced by stage.
- ``stored``: old-stage replay examples are exact examples selected from the
  stage's first train stream and reused later, balanced by stage.
- ``reservoir``: old-stage replay examples are sampled from one fixed-capacity
  reservoir over the observed stream.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import zoology.train as train_module
from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.utils import DataSegment
from zoology.experiments.rmt_mqar import class_incremental_ar_permuted_formal as formal
from zoology.experiments.rmt_mqar import class_incremental_ar_permuted_smoke as base
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism


TASK_SPECS = {
    "replay3_permuted_smoke": {
        "num_stages": 3,
        "train_examples": 512,
        "test_examples": 128,
        "max_epochs": 2,
    },
    "formal20_permuted": formal.FORMAL_TASK_SPECS["formal20_permuted"],
}


def _csv_env(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


SEEDS = _csv_env("PERMUTED_AR_REPLAY_SEEDS", [123, 456, 789], int)
TASK_NAMES = _csv_env("PERMUTED_AR_REPLAY_TASKS", ["formal20_permuted"], str)
MODEL_KEYS = _csv_env(
    "PERMUTED_AR_REPLAY_MODELS",
    ["attention", "base_rmt_nmem16"],
    str,
)
REPLAY_EXAMPLES_PER_OLD_STAGE = int(
    os.getenv("PERMUTED_AR_REPLAY_EXAMPLES_PER_OLD_STAGE", "64")
)
REPLAY_BUDGET_MODE = os.getenv("PERMUTED_AR_REPLAY_BUDGET_MODE", "fixed_total")
REPLAY_SOURCE_MODE = os.getenv("PERMUTED_AR_REPLAY_SOURCE_MODE", "oracle")
LR_SCHEDULER_MODE = os.getenv("PERMUTED_AR_REPLAY_LR_SCHEDULER_MODE", "stage_onecycle")
SLOW_UPDATE_MODE = os.getenv("PERMUTED_AR_REPLAY_SLOW_UPDATE_MODE", "accumulate")
EPOCH_EVAL_INTERVAL = int(os.getenv("PERMUTED_AR_REPLAY_EPOCH_EVAL_INTERVAL", "1"))

if REPLAY_BUDGET_MODE not in {"fixed_total", "replay_overhead"}:
    raise ValueError("PERMUTED_AR_REPLAY_BUDGET_MODE must be fixed_total or replay_overhead")
if REPLAY_SOURCE_MODE not in {"oracle", "stored", "reservoir"}:
    raise ValueError("PERMUTED_AR_REPLAY_SOURCE_MODE must be oracle, stored, or reservoir")

unknown_tasks = sorted(set(TASK_NAMES) - set(TASK_SPECS))
if unknown_tasks:
    raise ValueError(f"Unknown PERMUTED_AR_REPLAY_TASKS: {unknown_tasks}")

_MAX_SELECTED_STAGES = max(TASK_SPECS[task_name]["num_stages"] for task_name in TASK_NAMES)
REPLAY_BUFFER_CAPACITY = int(
    os.getenv(
        "PERMUTED_AR_REPLAY_BUFFER_CAPACITY",
        str(REPLAY_EXAMPLES_PER_OLD_STAGE * max(0, _MAX_SELECTED_STAGES - 1)),
    )
)


class BatchedTensorDataset(Dataset):
    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        slices: dict[str, Any] | None = None,
        shuffle_seed: int | None = None,
    ):
        if shuffle_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(shuffle_seed))
            order = torch.randperm(len(inputs), generator=generator)
            inputs = inputs[order]
            labels = labels[order]
        self.inputs = inputs
        self.labels = labels
        self.slices = slices or {}
        self.batch_size = batch_size
        self.num_examples = len(inputs)
        self.batches = list(range(0, self.num_examples, self.batch_size))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, batch_idx: int):
        start = self.batches[batch_idx]
        end = min(start + self.batch_size, self.num_examples)
        slc = slice(start, end)
        batch_size = end - start
        return self.inputs[slc], self.labels[slc], [self.slices] * batch_size


def _batch_sizes(config: ContinualTrainConfig):
    if isinstance(config.data.batch_size, int):
        return config.data.batch_size, config.data.batch_size
    return config.data.batch_size


def data_config(task_name: str, seed: int):
    spec = TASK_SPECS[task_name]
    num_stages = spec["num_stages"]
    return ContinualDataConfig(
        train_stage_configs=[
            base.stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=spec["train_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        test_stage_configs=[
            base.stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=spec["test_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        batch_size=base.BATCH_SIZE,
        seed=seed,
    )


def _segment_seed(config_seed: int, train_stage_idx: int, source_stage_idx: int, offset: int):
    return int(
        (
            config_seed * 1_000_003
            + train_stage_idx * 10_007
            + source_stage_idx * 1_009
            + offset
        )
        % (2**31 - 1)
    )


def _test_seed(config_seed: int, stage_idx: int):
    return _segment_seed(config_seed, stage_idx, stage_idx, 900_001)


def _build_test_dataloader(config: ContinualTrainConfig, stage_idx: int):
    _, test_batch_size = _batch_sizes(config)
    stage_config = config.data.test_stage_configs[stage_idx]
    segment = DataSegment.from_config(stage_config, seed=_test_seed(config.data.seed, stage_idx))
    dataset = BatchedTensorDataset(
        segment.inputs,
        segment.labels,
        batch_size=test_batch_size,
        slices=segment.slices,
        shuffle_seed=None,
    )
    return DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)


def _stage_replay_counts(
    base_examples: int,
    stage_idx: int,
    available_old_examples: int | None = None,
):
    desired_old_examples = REPLAY_EXAMPLES_PER_OLD_STAGE * stage_idx
    old_examples = (
        min(desired_old_examples, available_old_examples)
        if available_old_examples is not None
        else desired_old_examples
    )
    if REPLAY_BUDGET_MODE == "fixed_total":
        current_examples = base_examples - old_examples
        if current_examples <= 0:
            raise ValueError(
                "Fixed-total replay budget is exhausted: "
                f"base_examples={base_examples}, stage_idx={stage_idx}, "
                f"replay_examples_per_old_stage={REPLAY_EXAMPLES_PER_OLD_STAGE}."
            )
        total_examples = base_examples
    else:
        current_examples = base_examples
        total_examples = base_examples + old_examples
    return current_examples, old_examples, total_examples


def _select_stored_buffer(segment: DataSegment, stage_idx: int, seed: int):
    if REPLAY_EXAMPLES_PER_OLD_STAGE <= 0:
        return DataSegment(
            segment.inputs[:0].clone(),
            segment.labels[:0].clone(),
            slices=segment.slices,
        )
    if len(segment) < REPLAY_EXAMPLES_PER_OLD_STAGE:
        raise ValueError(
            "Cannot store more replay examples than current-stage examples: "
            f"stage_idx={stage_idx}, current_examples={len(segment)}, "
            f"replay_examples_per_old_stage={REPLAY_EXAMPLES_PER_OLD_STAGE}."
        )
    generator = torch.Generator()
    generator.manual_seed(_segment_seed(seed, stage_idx, stage_idx, 400_001))
    indices = torch.randperm(len(segment), generator=generator)[
        :REPLAY_EXAMPLES_PER_OLD_STAGE
    ]
    return DataSegment(
        segment.inputs[indices].clone(),
        segment.labels[indices].clone(),
        slices=segment.slices,
    )


class ReservoirReplayBuffer:
    """Fixed-size reservoir over exact observed training examples."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 0:
            raise ValueError("Reservoir replay capacity must be non-negative.")
        self.capacity = capacity
        self.generator = torch.Generator()
        self.generator.manual_seed(_segment_seed(seed, 0, 0, 700_001))
        self.inputs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.examples_seen = 0

    def __len__(self):
        return len(self.inputs)

    def add_segment(self, segment: DataSegment):
        if self.capacity == 0:
            self.examples_seen += len(segment)
            return
        for idx in range(len(segment)):
            self.examples_seen += 1
            input_item = segment.inputs[idx].detach().clone()
            label_item = segment.labels[idx].detach().clone()
            if len(self.inputs) < self.capacity:
                self.inputs.append(input_item)
                self.labels.append(label_item)
                continue
            replacement_idx = int(
                torch.randint(
                    low=0,
                    high=self.examples_seen,
                    size=(1,),
                    generator=self.generator,
                ).item()
            )
            if replacement_idx < self.capacity:
                self.inputs[replacement_idx] = input_item
                self.labels[replacement_idx] = label_item

    def sample(self, num_examples: int, seed: int) -> DataSegment:
        if num_examples <= 0 or len(self.inputs) == 0:
            if self.inputs:
                input_template = self.inputs[0]
                label_template = self.labels[0]
                return DataSegment(
                    input_template[:0].view(0, *input_template.shape).clone(),
                    label_template[:0].view(0, *label_template.shape).clone(),
                    slices={},
                )
            raise ValueError("Cannot infer empty reservoir tensor shapes before any data is observed.")
        if num_examples > len(self.inputs):
            raise ValueError(
                f"Requested {num_examples} reservoir examples, but buffer has {len(self.inputs)}."
            )
        generator = torch.Generator()
        generator.manual_seed(seed)
        indices = torch.randperm(len(self.inputs), generator=generator)[:num_examples]
        return DataSegment(
            torch.stack([self.inputs[int(idx)] for idx in indices], dim=0),
            torch.stack([self.labels[int(idx)] for idx in indices], dim=0),
            slices={},
        )


def _build_replay_train_dataloader(
    config: ContinualTrainConfig,
    stage_idx: int,
    stored_buffers: dict[int, DataSegment] | None = None,
    reservoir_buffer: ReservoirReplayBuffer | None = None,
):
    train_batch_size, _ = _batch_sizes(config)
    base_stage_config = config.data.train_stage_configs[stage_idx]
    base_examples = base_stage_config.num_examples
    available_old_examples = (
        len(reservoir_buffer)
        if REPLAY_SOURCE_MODE == "reservoir" and reservoir_buffer is not None
        else None
    )
    current_examples, old_examples, total_examples = _stage_replay_counts(
        base_examples,
        stage_idx,
        available_old_examples=available_old_examples,
    )

    segments = []
    stored_buffers = stored_buffers or {}
    if REPLAY_SOURCE_MODE == "stored":
        missing_buffers = [
            old_stage_idx
            for old_stage_idx in range(stage_idx)
            if old_stage_idx not in stored_buffers
        ]
        if missing_buffers:
            raise ValueError(f"Missing stored replay buffers: {missing_buffers}")
        segments.extend(stored_buffers[old_stage_idx] for old_stage_idx in range(stage_idx))
    elif REPLAY_SOURCE_MODE == "reservoir":
        if reservoir_buffer is None:
            raise ValueError("Reservoir replay requested without a reservoir buffer.")
        if old_examples > 0:
            segments.append(
                reservoir_buffer.sample(
                    old_examples,
                    seed=_segment_seed(config.data.seed, stage_idx, stage_idx, 600_001),
                )
            )
    else:
        for old_stage_idx in range(stage_idx):
            old_config = config.data.train_stage_configs[old_stage_idx].model_copy(
                update={"num_examples": REPLAY_EXAMPLES_PER_OLD_STAGE}
            )
            segments.append(
                DataSegment.from_config(
                    old_config,
                    seed=_segment_seed(config.data.seed, stage_idx, old_stage_idx, 100_001),
                )
            )

    current_config = base_stage_config.model_copy(update={"num_examples": current_examples})
    current_segment = DataSegment.from_config(
        current_config,
        seed=_segment_seed(config.data.seed, stage_idx, stage_idx, 200_001),
    )
    segments.append(current_segment)

    inputs = torch.cat([segment.inputs for segment in segments], dim=0)
    labels = torch.cat([segment.labels for segment in segments], dim=0)
    dataset = BatchedTensorDataset(
        inputs,
        labels,
        batch_size=train_batch_size,
        slices={},
        shuffle_seed=_segment_seed(config.data.seed, stage_idx, stage_idx, 300_001),
    )
    info = {
        "base_examples_per_stage": base_examples,
        "current_examples_per_epoch": current_examples,
        "old_examples_per_epoch": old_examples,
        "total_examples_per_epoch": total_examples,
        "replay_examples_per_old_stage": REPLAY_EXAMPLES_PER_OLD_STAGE,
        "replay_desired_old_examples_per_epoch": REPLAY_EXAMPLES_PER_OLD_STAGE * stage_idx,
        "replay_buffer_capacity": REPLAY_BUFFER_CAPACITY if REPLAY_SOURCE_MODE == "reservoir" else None,
        "replay_buffer_size_before": available_old_examples if available_old_examples is not None else None,
        "budget_mode": REPLAY_BUDGET_MODE,
        "source_mode": REPLAY_SOURCE_MODE,
    }
    stored_buffer = (
        _select_stored_buffer(current_segment, stage_idx, config.data.seed)
        if REPLAY_SOURCE_MODE == "stored"
        else None
    )
    return (
        DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False),
        info,
        stored_buffer,
        current_segment,
    )


def _dataloader_num_examples(dataloader: DataLoader):
    dataset = getattr(dataloader, "dataset", None)
    return getattr(dataset, "num_examples", None)


def _sequence_length_from_config(stage_config):
    return getattr(stage_config, "input_seq_len", None)


def _throughput_metrics(prefix: str, wall_seconds: float, examples: int = None, sequence_length: int = None):
    metrics = {f"{prefix}_wall_seconds": wall_seconds}
    if examples is not None:
        metrics[f"{prefix}_examples"] = examples
        if wall_seconds > 0:
            metrics[f"{prefix}_examples_per_second"] = examples / wall_seconds
    if examples is not None and sequence_length is not None:
        tokens = examples * sequence_length
        metrics[f"{prefix}_tokens"] = tokens
        if wall_seconds > 0:
            metrics[f"{prefix}_tokens_per_second"] = tokens / wall_seconds
    return metrics


def _stage_random_accuracy(config: ContinualTrainConfig, stage_idx: int):
    stage_config = config.data.test_stage_configs[stage_idx]
    random_accuracy = getattr(stage_config, "random_accuracy", None)
    if callable(random_accuracy):
        return float(random_accuracy())
    return None


def _train_log_interval(config: ContinualTrainConfig):
    value = os.getenv("ZOOLOGY_TRAIN_LOG_INTERVAL")
    if value is None:
        return config.train_log_interval
    return int(value)


def _sync_device_for_timing(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def train_continual_balanced_replay(config: ContinualTrainConfig, logger: base.CaptureLogger):
    set_determinism(config.seed)
    logger.log_config(config)

    if config.input_type != "discrete":
        raise ValueError("Balanced replay runner supports discrete input only.")

    model = LanguageModel(config.model)
    logger.log_model(model, config=config)

    stage_dataloaders = []
    replay_stage_info = []
    stored_buffers = {}
    reservoir_buffer = ReservoirReplayBuffer(REPLAY_BUFFER_CAPACITY, config.data.seed)
    for stage_idx in range(len(config.data.train_stage_configs)):
        train_dataloader, train_info, stored_buffer, current_segment = _build_replay_train_dataloader(
            config,
            stage_idx,
            stored_buffers=stored_buffers,
            reservoir_buffer=reservoir_buffer,
        )
        if stored_buffer is not None:
            stored_buffers[stage_idx] = stored_buffer
        if REPLAY_SOURCE_MODE == "reservoir":
            reservoir_buffer.add_segment(current_segment)
            train_info["replay_buffer_size_after"] = len(reservoir_buffer)
            train_info["replay_stream_examples_seen_after"] = reservoir_buffer.examples_seen
        test_dataloader = _build_test_dataloader(config, stage_idx)
        stage_dataloaders.append((train_dataloader, test_dataloader))
        replay_stage_info.append(train_info)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        train_dataloader=stage_dataloaders[0][0],
        test_dataloader=stage_dataloaders[0][1],
        input_type=config.input_type,
        max_epochs=config.max_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        early_stopping_metric=config.early_stopping_metric,
        early_stopping_threshold=config.early_stopping_threshold,
        train_log_interval=_train_log_interval(config),
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        slow_update_mode=getattr(config, "slow_update_mode", "skip"),
        device=device,
        logger=logger,
    )

    lr_scheduler_mode = getattr(config, "lr_scheduler_mode", "global_cosine")
    scheduler_epochs = (
        config.max_epochs * len(stage_dataloaders)
        if lr_scheduler_mode == "global_cosine"
        else config.max_epochs
    )
    scheduler_steps_per_epoch = None
    if lr_scheduler_mode == "stage_onecycle":
        scheduler_steps_per_epoch = max(1, trainer._optimizer_steps_per_epoch())
    trainer.initialize_training(
        scheduler_epochs=scheduler_epochs,
        scheduler_mode=lr_scheduler_mode,
        scheduler_steps_per_epoch=scheduler_steps_per_epoch,
    )

    best_stage_accuracy = {}
    learning_stage_accuracy = {}
    pre_learning_stage_accuracy = {}
    pretrain_stage_accuracy = {}
    stage_random_accuracy = {
        stage_idx: random_accuracy
        for stage_idx in range(len(stage_dataloaders))
        if (random_accuracy := _stage_random_accuracy(config, stage_idx)) is not None
    }

    if config.evaluate_future_stages:
        for eval_stage_idx in range(len(stage_dataloaders)):
            test_dataloader = stage_dataloaders[eval_stage_idx][1]
            pretrain_metrics = trainer.evaluate(
                test_dataloader,
                epoch_idx=0,
                metric_prefix=f"continual/pretrain/stage_{eval_stage_idx}",
                desc=f"Pretrain Continual Stage {eval_stage_idx}",
                log=False,
            )
            pretrain_stage_accuracy[eval_stage_idx] = pretrain_metrics[
                f"continual/pretrain/stage_{eval_stage_idx}/accuracy"
            ]
        pre_learning_stage_accuracy[0] = pretrain_stage_accuracy[0]

    cumulative_train_wall_seconds = 0.0
    cumulative_epoch_eval_wall_seconds = 0.0
    cumulative_seen_eval_wall_seconds = 0.0
    cumulative_wall_seconds = 0.0

    for stage_idx, (train_dataloader, _) in enumerate(stage_dataloaders):
        trainer.train_dataloader = train_dataloader
        trainer.log_context = {"continual/stage": stage_idx}
        if lr_scheduler_mode in {"stage_cosine", "stage_onecycle"}:
            trainer.reset_scheduler(
                scheduler_epochs=config.max_epochs,
                scheduler_mode=lr_scheduler_mode,
                scheduler_steps_per_epoch=max(
                    1,
                    trainer._optimizer_steps_per_epoch(train_dataloader),
                ),
                reset_lr=True,
            )

        if config.evaluate_future_stages and stage_idx not in pre_learning_stage_accuracy:
            test_dataloader = stage_dataloaders[stage_idx][1]
            pre_learning_metrics = trainer.evaluate(
                test_dataloader,
                epoch_idx=stage_idx * config.max_epochs,
                metric_prefix=f"continual/pre_learning/stage_{stage_idx}",
                desc=f"Pre-Learning Continual Stage {stage_idx}",
                log=False,
            )
            pre_learning_stage_accuracy[stage_idx] = pre_learning_metrics[
                f"continual/pre_learning/stage_{stage_idx}/accuracy"
            ]

        current_test_dataloader = stage_dataloaders[stage_idx][1]
        train_wall_seconds = 0.0
        epoch_eval_wall_seconds = 0.0
        train_epoch_summaries = []

        for local_epoch_idx in range(config.max_epochs):
            global_epoch_idx = stage_idx * config.max_epochs + local_epoch_idx
            epoch_log_context = {
                "continual/global_epoch": global_epoch_idx,
                "continual/local_epoch": local_epoch_idx,
                "continual/stage_epoch": local_epoch_idx + 1,
            }

            _sync_device_for_timing(device)
            epoch_train_wall_start = time.perf_counter()
            train_metrics = trainer.train_epoch(
                global_epoch_idx,
                desc=(
                    f"Train Replay Stage {stage_idx + 1}/{len(stage_dataloaders)} "
                    f"Epoch {local_epoch_idx + 1}/{config.max_epochs}"
                ),
                log_context=epoch_log_context,
            )
            if not getattr(trainer, "scheduler_steps_on_optimizer_step", False):
                trainer.scheduler.step()
            _sync_device_for_timing(device)
            epoch_train_wall = time.perf_counter() - epoch_train_wall_start
            train_wall_seconds += epoch_train_wall
            train_epoch_summaries.append(train_metrics)

            should_log_epoch_eval = (
                config.continual_epoch_eval_interval > 0
                and (
                    (local_epoch_idx + 1) % config.continual_epoch_eval_interval == 0
                    or local_epoch_idx == config.max_epochs - 1
                )
            )
            if should_log_epoch_eval:
                _sync_device_for_timing(device)
                epoch_eval_wall_start = time.perf_counter()
                current_epoch_metrics = trainer.evaluate(
                    current_test_dataloader,
                    epoch_idx=global_epoch_idx,
                    metric_prefix="continual/current_stage_epoch",
                    desc=(
                        f"Valid Replay Stage {stage_idx + 1}/{len(stage_dataloaders)} "
                        f"Epoch {local_epoch_idx + 1}/{config.max_epochs}"
                    ),
                    log=False,
                )
                _sync_device_for_timing(device)
                current_epoch_eval_wall = time.perf_counter() - epoch_eval_wall_start
                epoch_eval_wall_seconds += current_epoch_eval_wall
                logger.log({
                    "epoch": global_epoch_idx,
                    "continual/stage": stage_idx,
                    **epoch_log_context,
                    "continual/current_stage_epoch_index": local_epoch_idx,
                    "continual/current_stage_epoch_number": local_epoch_idx + 1,
                    "continual/current_stage_epoch_train_wall_seconds": epoch_train_wall,
                    "continual/current_stage_epoch_eval_wall_seconds": current_epoch_eval_wall,
                    f"continual/stage_{stage_idx}/epoch_accuracy": current_epoch_metrics[
                        "continual/current_stage_epoch/accuracy"
                    ],
                    f"continual/stage_{stage_idx}/epoch_loss": current_epoch_metrics[
                        "continual/current_stage_epoch/loss"
                    ],
                    **train_metrics,
                    **current_epoch_metrics,
                })

        stage_metrics = {}
        _sync_device_for_timing(device)
        seen_eval_wall_start = time.perf_counter()
        for eval_stage_idx in range(stage_idx + 1):
            test_dataloader = stage_dataloaders[eval_stage_idx][1]
            metrics = trainer.evaluate(
                test_dataloader,
                epoch_idx=(stage_idx + 1) * config.max_epochs - 1,
                metric_prefix=f"continual/stage_{eval_stage_idx}",
                desc=f"Valid Replay Stage {eval_stage_idx}/{stage_idx}",
                log=False,
            )
            stage_metrics[eval_stage_idx] = metrics
        _sync_device_for_timing(device)
        seen_eval_wall_seconds = time.perf_counter() - seen_eval_wall_start

        learning_stage_accuracy[stage_idx] = stage_metrics[stage_idx][
            f"continual/stage_{stage_idx}/accuracy"
        ]
        metrics = train_module._summarize_continual_eval(
            stage_idx=stage_idx,
            stage_metrics=stage_metrics,
            best_stage_accuracy=best_stage_accuracy,
            learning_stage_accuracy=learning_stage_accuracy,
            pre_learning_stage_accuracy=pre_learning_stage_accuracy,
            pretrain_stage_accuracy=pretrain_stage_accuracy,
            stage_random_accuracy=stage_random_accuracy,
        )

        train_examples_per_epoch = _dataloader_num_examples(train_dataloader)
        train_examples = (
            train_examples_per_epoch * config.max_epochs
            if train_examples_per_epoch is not None
            else None
        )
        train_sequence_length = _sequence_length_from_config(
            config.data.train_stage_configs[stage_idx]
        )
        stage_train_batches = len(train_dataloader) * config.max_epochs
        slow_update_freq = trainer._slow_update_freq()
        stage_optimizer_steps = (
            trainer._optimizer_steps_per_epoch(train_dataloader) * config.max_epochs
        )
        seen_eval_examples = 0
        seen_eval_sequence_tokens = 0
        stage_seen_eval_batches = 0
        for eval_stage_idx in range(stage_idx + 1):
            test_dataloader = stage_dataloaders[eval_stage_idx][1]
            stage_seen_eval_batches += len(test_dataloader)
            eval_examples = _dataloader_num_examples(test_dataloader)
            eval_sequence_length = _sequence_length_from_config(
                config.data.test_stage_configs[eval_stage_idx]
            )
            if eval_examples is not None:
                seen_eval_examples += eval_examples
                if eval_sequence_length is not None:
                    seen_eval_sequence_tokens += eval_examples * eval_sequence_length

        stage_epoch_loss = (
            float(np.mean([summary["train/epoch_loss"] for summary in train_epoch_summaries]))
            if train_epoch_summaries
            else float("nan")
        )
        stage_wall_seconds = (
            train_wall_seconds + epoch_eval_wall_seconds + seen_eval_wall_seconds
        )
        cumulative_train_wall_seconds += train_wall_seconds
        cumulative_epoch_eval_wall_seconds += epoch_eval_wall_seconds
        cumulative_seen_eval_wall_seconds += seen_eval_wall_seconds
        cumulative_wall_seconds += stage_wall_seconds

        timing_metrics = {
            **_throughput_metrics(
                "continual/stage_train",
                train_wall_seconds,
                examples=train_examples,
                sequence_length=train_sequence_length,
            ),
            **_throughput_metrics(
                "continual/stage_seen_eval",
                seen_eval_wall_seconds,
                examples=seen_eval_examples,
            ),
            "continual/stage_train_batches": stage_train_batches,
            "continual/stage_optimizer_steps": stage_optimizer_steps,
            "continual/slow_update_mode_is_accumulate": float(
                getattr(config, "slow_update_mode", "skip") == "accumulate"
            ),
            "continual/slow_update_freq": slow_update_freq,
            "continual/stage_seen_eval_batches": stage_seen_eval_batches,
            "continual/stage_seen_eval_tokens": seen_eval_sequence_tokens,
            "continual/lr": trainer.current_learning_rate(),
            "continual/stage_epoch_eval_wall_seconds": epoch_eval_wall_seconds,
            "continual/stage_train_epoch_loss": stage_epoch_loss,
            "continual/stage_wall_seconds": stage_wall_seconds,
            "continual/cumulative_train_wall_seconds": cumulative_train_wall_seconds,
            "continual/cumulative_epoch_eval_wall_seconds": cumulative_epoch_eval_wall_seconds,
            "continual/cumulative_seen_eval_wall_seconds": cumulative_seen_eval_wall_seconds,
            "continual/cumulative_wall_seconds": cumulative_wall_seconds,
            "continual/replay_examples_per_old_stage": REPLAY_EXAMPLES_PER_OLD_STAGE,
            "continual/replay_current_examples_per_epoch": replay_stage_info[stage_idx][
                "current_examples_per_epoch"
            ],
            "continual/replay_old_examples_per_epoch": replay_stage_info[stage_idx][
                "old_examples_per_epoch"
            ],
            "continual/replay_desired_old_examples_per_epoch": replay_stage_info[stage_idx][
                "replay_desired_old_examples_per_epoch"
            ],
            "continual/replay_total_examples_per_epoch": replay_stage_info[stage_idx][
                "total_examples_per_epoch"
            ],
            "continual/replay_budget_mode_is_fixed_total": float(
                REPLAY_BUDGET_MODE == "fixed_total"
            ),
            "continual/replay_source_mode_is_stored": float(
                REPLAY_SOURCE_MODE == "stored"
            ),
            "continual/replay_source_mode_is_reservoir": float(
                REPLAY_SOURCE_MODE == "reservoir"
            ),
        }
        if replay_stage_info[stage_idx].get("replay_buffer_capacity") is not None:
            timing_metrics.update(
                {
                    "continual/replay_buffer_capacity": replay_stage_info[stage_idx][
                        "replay_buffer_capacity"
                    ],
                    "continual/replay_buffer_size_before": replay_stage_info[stage_idx][
                        "replay_buffer_size_before"
                    ],
                    "continual/replay_buffer_size_after": replay_stage_info[stage_idx][
                        "replay_buffer_size_after"
                    ],
                    "continual/replay_stream_examples_seen_after": replay_stage_info[stage_idx][
                        "replay_stream_examples_seen_after"
                    ],
                }
            )
        if seen_eval_wall_seconds > 0:
            timing_metrics["continual/stage_seen_eval_tokens_per_second"] = (
                seen_eval_sequence_tokens / seen_eval_wall_seconds
            )

        stage_end_epoch = (stage_idx + 1) * config.max_epochs - 1
        logger.log({
            "epoch": stage_end_epoch,
            "continual/global_epoch": stage_end_epoch,
            "continual/local_epoch": config.max_epochs - 1,
            "continual/stage_epoch": config.max_epochs,
            **metrics,
            **timing_metrics,
        })

        if config.evaluate_future_stages and stage_idx + 1 < len(stage_dataloaders):
            next_stage_idx = stage_idx + 1
            test_dataloader = stage_dataloaders[next_stage_idx][1]
            next_pre_learning_metrics = trainer.evaluate(
                test_dataloader,
                epoch_idx=(stage_idx + 1) * config.max_epochs - 1,
                metric_prefix=f"continual/pre_learning/stage_{next_stage_idx}",
                desc=f"Pre-Learning Replay Stage {next_stage_idx}",
                log=False,
            )
            pre_learning_accuracy = next_pre_learning_metrics[
                f"continual/pre_learning/stage_{next_stage_idx}/accuracy"
            ]
            pre_learning_stage_accuracy[next_stage_idx] = pre_learning_accuracy
            logger.log({
                "epoch": stage_end_epoch,
                "continual/global_epoch": stage_end_epoch,
                "continual/local_epoch": config.max_epochs - 1,
                "continual/stage_epoch": config.max_epochs,
                "continual/stage": stage_idx,
                f"continual/stage_{next_stage_idx}/pre_learning_accuracy": pre_learning_accuracy,
                **(
                    {
                        f"continual/stage_{next_stage_idx}/fwt_from_random": (
                            pre_learning_accuracy - stage_random_accuracy[next_stage_idx]
                        )
                    }
                    if next_stage_idx in stage_random_accuracy
                    else {}
                ),
            })

    logger.finish()
    return replay_stage_info


def _model_settings(config: ContinualTrainConfig):
    return base._model_settings(config)


def _task_and_model_from_run_id(run_id: str):
    body = run_id.removeprefix("permuted-replay-").rsplit("-seed", 1)[0]
    for model_key in sorted(MODEL_KEYS, key=len, reverse=True):
        suffix = f"-{model_key}"
        if body.endswith(suffix):
            return body[: -len(suffix)], model_key
    raise ValueError(f"Cannot parse run_id: {run_id}")


def _final_summary(history: list[dict[str, Any]]):
    return base._final_summary(history)


def _stat(values: list[float]):
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_runs(runs: list[dict[str, Any]]):
    metric_map = {
        "final_seen_avg_accuracy": "continual/seen_avg_accuracy",
        "final_avg_learning_accuracy": "continual/avg_learning_accuracy",
        "final_plasticity": "continual/plasticity",
        "final_avg_bwt": "continual/avg_bwt",
        "final_avg_forgetting_from_learning": "continual/avg_forgetting_from_learning",
        "final_old_stage_avg_accuracy": "continual/old_stage_avg_accuracy",
        "final_old_stage_avg_forgetting": "continual/old_stage_avg_forgetting",
        "final_old_stage_avg_forgetting_from_learning": "continual/old_stage_avg_forgetting_from_learning",
        "final_old_stage_avg_bwt": "continual/old_stage_avg_bwt",
        "seen_avg_accuracy_stage_auc": "continual/seen_avg_accuracy_stage_auc",
        "old_stage_avg_accuracy_stage_auc": "continual/old_stage_avg_accuracy_stage_auc",
        "old_stage_avg_forgetting_stage_auc": "continual/old_stage_avg_forgetting_stage_auc",
        "old_stage_avg_forgetting_from_learning_stage_auc": "continual/old_stage_avg_forgetting_from_learning_stage_auc",
        "old_stage_avg_bwt_stage_auc": "continual/old_stage_avg_bwt_stage_auc",
        "final_cumulative_wall_seconds": "continual/cumulative_wall_seconds",
    }
    grouped = {}
    for run in runs:
        key = (
            f"{run['task_name']}::{run['model_key']}::"
            f"{run['replay_source_mode']}_replay"
            f"{run['replay_examples_per_old_stage']}::"
            f"buffer{run.get('replay_buffer_capacity')}::"
            f"{run['replay_budget_mode']}"
        )
        grouped.setdefault(key, []).append(run)

    aggregates = {}
    for key, group in grouped.items():
        metrics = {}
        for out_name, source_name in metric_map.items():
            values = [
                run["summary"][source_name]
                for run in group
                if source_name in run["summary"]
                and isinstance(run["summary"][source_name], (int, float))
            ]
            if values:
                metrics[out_name] = _stat(values)
        first = group[0]
        aggregates[key] = {
            "num_runs": len(group),
            "task_spec": TASK_SPECS[first["task_name"]],
            "model_name": first["model_name"],
            "model_info": first["model_info"],
            "model_settings": first["model_settings"],
            "replay_examples_per_old_stage": first["replay_examples_per_old_stage"],
            "replay_buffer_capacity": first.get("replay_buffer_capacity"),
            "replay_budget_mode": first["replay_budget_mode"],
            "replay_source_mode": first["replay_source_mode"],
            "metrics": metrics,
        }
    return aggregates


configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=formal.model_by_key(model_key),
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "permuted_mapping",
                "replay",
                (
                    "reservoir_replay"
                    if REPLAY_SOURCE_MODE == "reservoir"
                    else "balanced_replay"
                ),
                f"{REPLAY_SOURCE_MODE}_replay",
                task_name,
            ]
        ),
        max_epochs=TASK_SPECS[task_name]["max_epochs"],
        learning_rate=base.LEARNING_RATE,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        train_log_interval=0,
        continual_epoch_eval_interval=EPOCH_EVAL_INTERVAL,
        lr_scheduler_mode=LR_SCHEDULER_MODE,
        slow_update_mode=SLOW_UPDATE_MODE,
        slice_keys=[
            "stage_idx",
            "eval_mode",
            "associations_per_stage",
            "num_query_associations",
            "input_seq_len",
            "value_mapping",
            "association_table_seed",
        ],
        seed=seed,
        run_id=f"permuted-replay-{task_name}-{model_key}-seed{seed}",
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for model_key in MODEL_KEYS
]


def run_all(output_prefix: str = "class_incremental_ar_permuted_replay"):
    original_tqdm = train_module.tqdm
    train_module.tqdm = base.QuietTqdm

    runs = []
    try:
        total = len(configs)
        for idx, config in enumerate(configs, start=1):
            base.CaptureLogger.instances.clear()
            logger = base.CaptureLogger(config)
            task_name, model_key = _task_and_model_from_run_id(config.run_id)
            print(f"[{idx}/{total}] {config.run_id}")
            replay_stage_info = train_continual_balanced_replay(config, logger)
            summary = _final_summary(logger.history)
            task_spec = TASK_SPECS[task_name]
            run = {
                "run_id": config.run_id,
                "task_name": task_name,
                "model_key": model_key,
                "model_name": config.model.name,
                "seed": config.seed,
                "model_settings": _model_settings(config),
                "model_info": logger.model_info,
                "lr_scheduler_mode": getattr(config, "lr_scheduler_mode", "global_cosine"),
                "slow_update_mode": getattr(config, "slow_update_mode", "skip"),
                "continual_epoch_eval_interval": getattr(config, "continual_epoch_eval_interval", 0),
                "replay_examples_per_old_stage": REPLAY_EXAMPLES_PER_OLD_STAGE,
                "replay_buffer_capacity": (
                    REPLAY_BUFFER_CAPACITY if REPLAY_SOURCE_MODE == "reservoir" else None
                ),
                "replay_budget_mode": REPLAY_BUDGET_MODE,
                "replay_source_mode": REPLAY_SOURCE_MODE,
                "replay_stage_info": replay_stage_info,
                "summary": summary,
                "history": base._json_safe_history(logger.history),
                "stage_end_history": base._stage_end_history(logger.history),
                "stage_end_matrices": base._stage_end_matrices(
                    logger.history,
                    num_stages=task_spec["num_stages"],
                ),
                "current_stage_epoch_history": base._current_stage_epoch_history(logger.history),
                "current_stage_epoch_curves": base._current_stage_epoch_curves(logger.history),
            }
            runs.append(run)
            print(
                "    "
                f"seen={summary.get('continual/seen_avg_accuracy', float('nan')):.4f} "
                f"learn={summary.get('continual/avg_learning_accuracy', float('nan')):.4f} "
                f"plast={summary.get('continual/plasticity', float('nan')):.4f} "
                f"bwt={summary.get('continual/avg_bwt', float('nan')):.4f} "
                f"forget={summary.get('continual/avg_forgetting_from_learning', float('nan')):.4f} "
                f"total={summary.get('continual/cumulative_wall_seconds', float('nan')):.1f}s"
            )
    finally:
        train_module.tqdm = original_tqdm

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_slug = "_".join(TASK_NAMES)
    replay_slug = f"replay{REPLAY_EXAMPLES_PER_OLD_STAGE}"
    if REPLAY_SOURCE_MODE == "reservoir":
        replay_slug = f"{replay_slug}_buffer{REPLAY_BUFFER_CAPACITY}"
    output_path = (
        Path("results")
        / (
            f"{output_prefix}_{LR_SCHEDULER_MODE}_{SLOW_UPDATE_MODE}_"
            f"{REPLAY_BUDGET_MODE}_{REPLAY_SOURCE_MODE}_"
            f"{replay_slug}_"
            f"{task_slug}_{created_at}.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "experiment": "replay for formal permuted class-incremental AR",
        "module": __name__,
        "setup": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tasks": TASK_NAMES,
            "models": MODEL_KEYS,
            "seeds": SEEDS,
            "replay_examples_per_old_stage": REPLAY_EXAMPLES_PER_OLD_STAGE,
            "replay_buffer_capacity": (
                REPLAY_BUFFER_CAPACITY if REPLAY_SOURCE_MODE == "reservoir" else None
            ),
            "replay_budget_mode": REPLAY_BUDGET_MODE,
            "replay_source_mode": REPLAY_SOURCE_MODE,
            "vocab_size": base.VOCAB_SIZE,
            "associations_per_stage": base.ASSOCIATIONS_PER_STAGE,
            "num_query_associations": base.NUM_QUERY_ASSOCIATIONS,
            "input_seq_len": base.INPUT_SEQ_LEN,
            "batch_size": base.BATCH_SIZE,
            "segment_len": base.SEGMENT_LEN,
            "learning_rate": base.LEARNING_RATE,
            "lr_scheduler_mode": LR_SCHEDULER_MODE,
            "slow_update_mode": SLOW_UPDATE_MODE,
            "continual_epoch_eval_interval": EPOCH_EVAL_INTERVAL,
            "value_mapping": base.VALUE_MAPPING,
            "association_table_seed": base.ASSOCIATION_TABLE_SEED,
            "stage_local_random_accuracy": 1.0 / base.ASSOCIATIONS_PER_STAGE,
            "raw_metric_history_saved": True,
            "stage_end_matrices_saved": True,
            "current_stage_epoch_curves_saved": True,
        },
        "task_specs": {name: TASK_SPECS[name] for name in TASK_NAMES},
        "aggregates": aggregate_runs(runs),
        "runs": runs,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"WROTE {output_path}")
    print("AGGREGATES")
    for key, aggregate in sorted(payload["aggregates"].items()):
        metrics = aggregate["metrics"]
        seen = metrics.get("final_seen_avg_accuracy", {})
        learn = metrics.get("final_avg_learning_accuracy", {})
        plast = metrics.get("final_plasticity", {})
        bwt = metrics.get("final_avg_bwt", {})
        forget = metrics.get("final_avg_forgetting_from_learning", {})
        wall = metrics.get("final_cumulative_wall_seconds", {})
        print(
            f"  {key} "
            f"seen={seen.get('mean', float('nan')):.4f}+/-{seen.get('std', float('nan')):.4f} "
            f"learn={learn.get('mean', float('nan')):.4f}+/-{learn.get('std', float('nan')):.4f} "
            f"plast={plast.get('mean', float('nan')):.4f}+/-{plast.get('std', float('nan')):.4f} "
            f"bwt={bwt.get('mean', float('nan')):.4f}+/-{bwt.get('std', float('nan')):.4f} "
            f"forget={forget.get('mean', float('nan')):.4f}+/-{forget.get('std', float('nan')):.4f} "
            f"total={wall.get('mean', float('nan')):.1f}+/-{wall.get('std', float('nan')):.1f}s"
        )
    return output_path


if __name__ == "__main__":
    run_all()
