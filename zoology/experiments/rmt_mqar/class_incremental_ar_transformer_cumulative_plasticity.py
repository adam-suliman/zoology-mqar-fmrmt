"""CIFAR-like cumulative-replay AR plasticity probe for a Transformer.

At stage t the continuous model trains on examples sampled from all seen
association stages 0..t. Plasticity is measured with a current-stage probe and a
fresh same-exposure Transformer trained only on the current stage.

Example smoke:

```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \
  --num-stages 3 \
  --seeds 123 \
  --examples-per-seen-stage 256 \
  --test-examples-per-stage 128 \
  --epochs-per-stage 2 \
  --output-prefix transformer_cumulative_ar_plasticity_smoke
```
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
import torch

import zoology.train as train_module
from zoology.config import DataConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.data.utils import prepare_data
from zoology.experiments.rmt_mqar._common import transformer_model
from zoology.model import LanguageModel
from zoology.train import Trainer
from zoology.utils import set_determinism


ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.1
ASSOCIATION_TABLE_SEED = 20260522
VALUE_MAPPING = "permuted"
LOSS_FRESH_THRESHOLD = 0.95
LOSS_CONTINUOUS_THRESHOLD = 0.80
LOSS_GAP_THRESHOLD = 0.20
LOSS_CONSECUTIVE_STAGES = 2


@dataclass
class ExperimentArgs:
    num_stages: int
    seeds: list[int]
    examples_per_seen_stage: int
    test_examples_per_stage: int
    epochs_per_stage: int
    output_prefix: str
    vocab_size: int | None = None
    batch_size: int = BATCH_SIZE
    input_seq_len: int = INPUT_SEQ_LEN
    associations_per_stage: int = ASSOCIATIONS_PER_STAGE
    num_query_associations: int = NUM_QUERY_ASSOCIATIONS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    association_table_seed: int = ASSOCIATION_TABLE_SEED
    value_mapping: str = VALUE_MAPPING
    results_dir: str = "results"
    verbose_progress: bool = False


class CaptureLogger:
    def __init__(self):
        self.history = []

    def log(self, metrics):
        self.history.append(dict(metrics))


class QuietTqdm:
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = iterable
        self.total = total

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


def parse_seeds(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def max_stages_for_vocab(vocab_size: int, associations_per_stage: int) -> int:
    key_vocab_size = vocab_size // 2
    return (key_vocab_size - 1) // associations_per_stage


def default_vocab_size(num_stages: int, associations_per_stage: int) -> int:
    if num_stages <= max_stages_for_vocab(1024, associations_per_stage):
        return 1024
    if num_stages <= max_stages_for_vocab(2048, associations_per_stage):
        return 2048
    raise ValueError(
        f"{num_stages} stages with {associations_per_stage} associations/stage "
        "do not fit in vocab_size 2048."
    )


def resolved_vocab_size(args: ExperimentArgs) -> int:
    vocab_size = args.vocab_size
    if vocab_size is None:
        vocab_size = default_vocab_size(args.num_stages, args.associations_per_stage)
    max_stages = max_stages_for_vocab(vocab_size, args.associations_per_stage)
    if args.num_stages > max_stages:
        raise ValueError(
            f"num_stages={args.num_stages} does not fit vocab_size={vocab_size}; "
            f"max supported stages is {max_stages}. Use a larger --vocab-size."
        )
    return vocab_size


def stage_config(
    args: ExperimentArgs,
    stage_idx: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
) -> ClassIncrementalARConfig:
    return ClassIncrementalARConfig(
        stage_idx=stage_idx,
        num_stages=args.num_stages,
        associations_per_stage=args.associations_per_stage,
        num_query_associations=args.num_query_associations,
        input_seq_len=args.input_seq_len,
        vocab_size=resolved_vocab_size(args),
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        value_mapping=args.value_mapping,
        association_table_seed=args.association_table_seed,
        include_slices=True,
    )


def cumulative_train_config(args: ExperimentArgs, stage_idx: int) -> ClassIncrementalARConfig:
    return stage_config(
        args,
        stage_idx=stage_idx,
        num_examples=args.examples_per_seen_stage * (stage_idx + 1),
        eval_mode="seen",
        distractor_mode="seen",
    )


def cumulative_test_config(args: ExperimentArgs, stage_idx: int) -> ClassIncrementalARConfig:
    return stage_config(
        args,
        stage_idx=stage_idx,
        num_examples=args.test_examples_per_stage * (stage_idx + 1),
        eval_mode="seen",
        distractor_mode="seen",
    )


def current_train_config(args: ExperimentArgs, stage_idx: int) -> ClassIncrementalARConfig:
    return stage_config(
        args,
        stage_idx=stage_idx,
        num_examples=args.examples_per_seen_stage,
        eval_mode="current",
        distractor_mode="seen",
    )


def current_test_config(args: ExperimentArgs, stage_idx: int) -> ClassIncrementalARConfig:
    return stage_config(
        args,
        stage_idx=stage_idx,
        num_examples=args.test_examples_per_stage,
        eval_mode="current",
        distractor_mode="seen",
    )


def build_dataloaders(
    train_config: ClassIncrementalARConfig,
    test_config: ClassIncrementalARConfig,
    batch_size: int,
    seed: int,
):
    return tuple(
        prepare_data(
            DataConfig(
                train_configs=[train_config],
                test_configs=[test_config],
                batch_size=batch_size,
                seed=seed,
            )
        )
    )


def make_model(args: ExperimentArgs) -> LanguageModel:
    return LanguageModel(
        transformer_model(
            input_seq_len=args.input_seq_len,
            vocab_size=resolved_vocab_size(args),
        )
    )


def make_trainer(
    args: ExperimentArgs,
    model: LanguageModel,
    train_dataloader,
    test_dataloader,
    device: str,
    scheduler_epochs: int,
) -> Trainer:
    logger = CaptureLogger()
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        input_type="discrete",
        max_epochs=args.epochs_per_stage,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_metric=None,
        early_stopping_threshold=None,
        train_log_interval=0,
        slice_keys=[
            "stage_idx",
            "eval_mode",
            "distractor_mode",
            "value_mapping",
            "association_table_seed",
        ],
        loss_type="ce",
        device=device,
        logger=logger,
    )
    trainer.initialize_training(scheduler_epochs=scheduler_epochs)
    return trainer


def evaluate_accuracy(
    trainer: Trainer,
    dataloader,
    epoch_idx: int,
    metric_prefix: str,
) -> float:
    metrics = trainer.evaluate(
        dataloader,
        epoch_idx=epoch_idx,
        metric_prefix=metric_prefix,
        desc=metric_prefix,
        log=False,
    )
    return float(metrics[f"{metric_prefix}/accuracy"])


def learning_auc(epoch_accuracies: list[float]) -> float:
    return float(mean(epoch_accuracies)) if epoch_accuracies else 0.0


def stage_has_plasticity_loss(
    stage: dict[str, Any],
    fresh_threshold: float = LOSS_FRESH_THRESHOLD,
    continuous_threshold: float = LOSS_CONTINUOUS_THRESHOLD,
    gap_threshold: float = LOSS_GAP_THRESHOLD,
) -> bool:
    fresh_final = stage["fresh"]["current_final_accuracy"]
    continuous_final = stage["continuous"]["current_final_accuracy"]
    gap_final = stage["plasticity_gap_final"]
    return bool(
        fresh_final >= fresh_threshold
        and (continuous_final <= continuous_threshold or gap_final >= gap_threshold)
    )


def detect_first_plasticity_loss_stage(
    stages: list[dict[str, Any]],
    consecutive_stages: int = LOSS_CONSECUTIVE_STAGES,
) -> dict[str, Any]:
    flags = [stage_has_plasticity_loss(stage) for stage in stages]
    required = min(consecutive_stages, len(flags))
    first_stage = None
    if required > 0:
        for start_idx in range(0, len(flags) - required + 1):
            if all(flags[start_idx : start_idx + required]):
                first_stage = stages[start_idx]["stage_idx"]
                break
    return {
        "first_plasticity_loss_stage": first_stage,
        "stage_flags": flags,
        "fresh_threshold": LOSS_FRESH_THRESHOLD,
        "continuous_threshold": LOSS_CONTINUOUS_THRESHOLD,
        "gap_threshold": LOSS_GAP_THRESHOLD,
        "consecutive_stages": consecutive_stages,
    }


def _examples_tokens(num_examples: int, epochs: int, sequence_length: int) -> dict[str, int]:
    examples = num_examples * epochs
    return {
        "examples": examples,
        "tokens": examples * sequence_length,
    }


def run_fresh_probe(
    args: ExperimentArgs,
    stage_idx: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    set_determinism(seed + 100_000 + stage_idx)
    train_dl, current_test_dl = build_dataloaders(
        current_train_config(args, stage_idx),
        current_test_config(args, stage_idx),
        batch_size=args.batch_size,
        seed=seed + 20_000 + stage_idx,
    )
    model = make_model(args)
    trainer = make_trainer(
        args,
        model=model,
        train_dataloader=train_dl,
        test_dataloader=current_test_dl,
        device=device,
        scheduler_epochs=args.epochs_per_stage,
    )
    pre_accuracy = evaluate_accuracy(
        trainer,
        current_test_dl,
        epoch_idx=0,
        metric_prefix=f"fresh/stage_{stage_idx}/pre",
    )
    epoch_accuracies = []
    train_start = time.perf_counter()
    for local_epoch_idx in range(args.epochs_per_stage):
        trainer.train_epoch(
            local_epoch_idx,
            desc=f"Fresh Stage {stage_idx + 1}/{args.num_stages} Epoch {local_epoch_idx + 1}/{args.epochs_per_stage}",
        )
        trainer.scheduler.step()
        accuracy = evaluate_accuracy(
            trainer,
            current_test_dl,
            epoch_idx=local_epoch_idx,
            metric_prefix=f"fresh/stage_{stage_idx}/epoch_{local_epoch_idx}",
        )
        epoch_accuracies.append(accuracy)
    train_wall_seconds = time.perf_counter() - train_start
    train_counts = _examples_tokens(
        args.examples_per_seen_stage,
        args.epochs_per_stage,
        args.input_seq_len,
    )
    return {
        "current_pre_accuracy": pre_accuracy,
        "current_epoch_accuracy": epoch_accuracies,
        "current_final_accuracy": epoch_accuracies[-1],
        "current_learning_auc": learning_auc(epoch_accuracies),
        "train_wall_seconds": train_wall_seconds,
        "train_examples": train_counts["examples"],
        "train_tokens": train_counts["tokens"],
    }


def run_seed(args: ExperimentArgs, seed: int) -> dict[str, Any]:
    set_determinism(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initial_train_dl, initial_test_dl = build_dataloaders(
        cumulative_train_config(args, 0),
        current_test_config(args, 0),
        batch_size=args.batch_size,
        seed=seed + 10_000,
    )
    model = make_model(args)
    trainer = make_trainer(
        args,
        model=model,
        train_dataloader=initial_train_dl,
        test_dataloader=initial_test_dl,
        device=device,
        scheduler_epochs=args.epochs_per_stage * args.num_stages,
    )

    stages = []
    model_info = {
        "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_size": model.state_size(sequence_length=args.input_seq_len),
    }

    for stage_idx in range(args.num_stages):
        print(f"[seed {seed}] cumulative stage {stage_idx + 1}/{args.num_stages}")
        train_dl, current_test_dl = build_dataloaders(
            cumulative_train_config(args, stage_idx),
            current_test_config(args, stage_idx),
            batch_size=args.batch_size,
            seed=seed + 10_000 + stage_idx,
        )
        _, cumulative_test_dl = build_dataloaders(
            cumulative_train_config(args, stage_idx),
            cumulative_test_config(args, stage_idx),
            batch_size=args.batch_size,
            seed=seed + 30_000 + stage_idx,
        )
        trainer.train_dataloader = train_dl
        trainer.test_dataloader = current_test_dl

        global_epoch_start = stage_idx * args.epochs_per_stage
        current_pre_accuracy = evaluate_accuracy(
            trainer,
            current_test_dl,
            epoch_idx=global_epoch_start,
            metric_prefix=f"continuous/stage_{stage_idx}/pre",
        )

        epoch_accuracies = []
        train_start = time.perf_counter()
        for local_epoch_idx in range(args.epochs_per_stage):
            global_epoch_idx = global_epoch_start + local_epoch_idx
            trainer.train_epoch(
                global_epoch_idx,
                desc=(
                    f"Continuous Stage {stage_idx + 1}/{args.num_stages} "
                    f"Epoch {local_epoch_idx + 1}/{args.epochs_per_stage}"
                ),
            )
            trainer.scheduler.step()
            accuracy = evaluate_accuracy(
                trainer,
                current_test_dl,
                epoch_idx=global_epoch_idx,
                metric_prefix=f"continuous/stage_{stage_idx}/epoch_{local_epoch_idx}",
            )
            epoch_accuracies.append(accuracy)
        train_wall_seconds = time.perf_counter() - train_start

        cumulative_seen_accuracy = evaluate_accuracy(
            trainer,
            cumulative_test_dl,
            epoch_idx=global_epoch_start + args.epochs_per_stage - 1,
            metric_prefix=f"continuous/stage_{stage_idx}/cumulative_seen",
        )

        stage_local_row = {}
        for eval_stage_idx in range(stage_idx + 1):
            _, local_test_dl = build_dataloaders(
                current_train_config(args, eval_stage_idx),
                current_test_config(args, eval_stage_idx),
                batch_size=args.batch_size,
                seed=seed + 40_000 + stage_idx * args.num_stages + eval_stage_idx,
            )
            stage_local_row[str(eval_stage_idx)] = evaluate_accuracy(
                trainer,
                local_test_dl,
                epoch_idx=global_epoch_start + args.epochs_per_stage - 1,
                metric_prefix=f"continuous/stage_{stage_idx}/local_eval_{eval_stage_idx}",
            )

        fresh = run_fresh_probe(args, stage_idx, seed, device)
        continuous_counts = _examples_tokens(
            args.examples_per_seen_stage * (stage_idx + 1),
            args.epochs_per_stage,
            args.input_seq_len,
        )
        continuous = {
            "current_pre_accuracy": current_pre_accuracy,
            "current_epoch_accuracy": epoch_accuracies,
            "current_final_accuracy": epoch_accuracies[-1],
            "current_learning_auc": learning_auc(epoch_accuracies),
            "cumulative_seen_accuracy": cumulative_seen_accuracy,
            "stage_local_accuracy_row": stage_local_row,
            "train_wall_seconds": train_wall_seconds,
            "train_examples": continuous_counts["examples"],
            "train_tokens": continuous_counts["tokens"],
        }

        stage_record = {
            "stage_idx": stage_idx,
            "continuous": continuous,
            "fresh": fresh,
            "plasticity_gap_final": (
                fresh["current_final_accuracy"] - continuous["current_final_accuracy"]
            ),
            "plasticity_gap_auc": (
                fresh["current_learning_auc"] - continuous["current_learning_auc"]
            ),
        }
        stages.append(stage_record)

    return {
        "seed": seed,
        "device": device,
        "model_info": model_info,
        "stages": stages,
        "detection": detect_first_plasticity_loss_stage(stages),
    }


def _stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_stage = {}
    metric_paths = {
        "continuous_current_final_accuracy": ("continuous", "current_final_accuracy"),
        "continuous_current_learning_auc": ("continuous", "current_learning_auc"),
        "fresh_current_final_accuracy": ("fresh", "current_final_accuracy"),
        "fresh_current_learning_auc": ("fresh", "current_learning_auc"),
        "plasticity_gap_final": ("plasticity_gap_final",),
        "plasticity_gap_auc": ("plasticity_gap_auc",),
        "continuous_cumulative_seen_accuracy": ("continuous", "cumulative_seen_accuracy"),
    }
    for stage_idx in range(max(len(run["stages"]) for run in runs)):
        stage_runs = [run["stages"][stage_idx] for run in runs if stage_idx < len(run["stages"])]
        stage_summary = {}
        for metric_name, path in metric_paths.items():
            values = []
            for stage in stage_runs:
                value = stage
                for key in path:
                    value = value[key]
                values.append(float(value))
            stage_summary[metric_name] = _stat(values)
        per_stage[str(stage_idx)] = stage_summary
    return {
        "per_stage": per_stage,
        "first_plasticity_loss_stage_by_seed": {
            str(run["seed"]): run["detection"]["first_plasticity_loss_stage"]
            for run in runs
        },
    }


def write_launcher_script(path: Path):
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

# Smoke
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \\
  --num-stages 3 \\
  --seeds 123 \\
  --examples-per-seen-stage 256 \\
  --test-examples-per-stage 128 \\
  --epochs-per-stage 2 \\
  --output-prefix transformer_cumulative_ar_plasticity_smoke

# Full 20-stage run
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \\
  --num-stages 20 \\
  --seeds 123,456,789 \\
  --examples-per-seen-stage 1024 \\
  --test-examples-per-stage 512 \\
  --epochs-per-stage 8 \\
  --output-prefix transformer_cumulative_ar_plasticity_20stage

# Escalation if no loss appears in the 20-stage run
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \\
  --num-stages 50 \\
  --seeds 123 \\
  --examples-per-seen-stage 1024 \\
  --test-examples-per-stage 512 \\
  --epochs-per-stage 8 \\
  --vocab-size 2048 \\
  --output-prefix transformer_cumulative_ar_plasticity_50stage_seed123
"""
    )
    path.chmod(0o755)


def run_experiment(args: ExperimentArgs) -> dict[str, Any]:
    args.vocab_size = resolved_vocab_size(args)
    if not args.verbose_progress:
        train_module.tqdm = QuietTqdm
    print(
        f"Running Transformer cumulative AR plasticity probe: "
        f"stages={args.num_stages} seeds={args.seeds} vocab={args.vocab_size}"
    )
    runs = [run_seed(args, seed) for seed in args.seeds]
    payload = {
        "experiment": "transformer_cumulative_ar_plasticity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "setup": asdict(args),
        "runs": runs,
        "aggregates": aggregate_runs(runs),
        "notes": [
            "Cumulative replay is a plasticity/intransigence probe, not a no-replay forgetting benchmark.",
            "Fresh probes train on current-stage examples_per_seen_stage, matching the continuous stream's effective new-stage exposure.",
            "Old-stage changes under cumulative replay should be interpreted as stability/interference, not forgetting.",
        ],
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{args.output_prefix}_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_launcher_script(Path("run_transformer_cumulative_ar_plasticity.sh"))
    print(f"WROTE {output_path}")
    print("DETECTION")
    for run in runs:
        print(
            f"  seed {run['seed']}: "
            f"{run['detection']['first_plasticity_loss_stage']}"
        )
    return payload


def parse_args(argv: list[str] | None = None) -> ExperimentArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-stages", type=int, required=True)
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--examples-per-seen-stage", type=int, required=True)
    parser.add_argument("--test-examples-per-stage", type=int, required=True)
    parser.add_argument("--epochs-per-stage", type=int, required=True)
    parser.add_argument("--output-prefix", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--input-seq-len", type=int, default=INPUT_SEQ_LEN)
    parser.add_argument("--associations-per-stage", type=int, default=ASSOCIATIONS_PER_STAGE)
    parser.add_argument("--num-query-associations", type=int, default=NUM_QUERY_ASSOCIATIONS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--association-table-seed", type=int, default=ASSOCIATION_TABLE_SEED)
    parser.add_argument("--value-mapping", type=str, default=VALUE_MAPPING, choices=["aligned", "permuted"])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--verbose-progress", action="store_true")
    ns = parser.parse_args(argv)
    args = ExperimentArgs(**vars(ns))
    args.vocab_size = resolved_vocab_size(args)
    return args


def main(argv: list[str] | None = None):
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
