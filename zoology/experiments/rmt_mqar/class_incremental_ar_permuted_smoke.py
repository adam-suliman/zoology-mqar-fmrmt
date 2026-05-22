"""Short smoke experiments for per-stage random-permutation class-incremental AR.

The task keeps a fixed random key->value permutation per stage, shared across
train/test generation seeds through ``association_table_seed``. This removes the
aligned key-offset -> value-offset shortcut while preserving stable stage
knowledge for CL metrics.

Default local run is intentionally small:

- one-stage permuted solvability check
- 3-stage permuted continual smoke
- seeds 123,456
- Transformer, Base RMT n_mem=4, stable FMRMT

Run locally and save JSON:

``.venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_permuted_smoke``

Narrow with env vars:

``PERMUTED_AR_SMOKE_TASKS=one_stage_permuted``
``PERMUTED_AR_SMOKE_MODELS=base_rmt_nmem4,fmrmt_stable``
``PERMUTED_AR_SMOKE_SEEDS=123``
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.experiments.rmt_mqar._common import (
    base_rmt_model,
    fast_memory_rmt_model,
    transformer_model,
)


VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
BATCH_SIZE = 64
SEGMENT_LEN = 64
LEARNING_RATE = 3e-3
ASSOCIATION_TABLE_SEED = 20260522
VALUE_MAPPING = "permuted"

TASK_SPECS = {
    "one_stage_permuted": {
        "num_stages": 1,
        "train_examples": 1024,
        "test_examples": 256,
        "max_epochs": 8,
    },
    "continual3_permuted_smoke": {
        "num_stages": 3,
        "train_examples": 512,
        "test_examples": 256,
        "max_epochs": 4,
    },
}


def _csv_env(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


SEEDS = _csv_env("PERMUTED_AR_SMOKE_SEEDS", [123, 456], int)
TASK_NAMES = _csv_env("PERMUTED_AR_SMOKE_TASKS", TASK_SPECS.keys(), str)
MODEL_KEYS = _csv_env(
    "PERMUTED_AR_SMOKE_MODELS",
    ["attention", "base_rmt_nmem4", "fmrmt_stable"],
    str,
)

unknown_tasks = sorted(set(TASK_NAMES) - set(TASK_SPECS))
if unknown_tasks:
    raise ValueError(f"Unknown PERMUTED_AR_SMOKE_TASKS: {unknown_tasks}")


def stage_config(
    stage_idx: int,
    num_stages: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
):
    return ClassIncrementalARConfig(
        stage_idx=stage_idx,
        num_stages=num_stages,
        associations_per_stage=ASSOCIATIONS_PER_STAGE,
        num_query_associations=NUM_QUERY_ASSOCIATIONS,
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        value_mapping=VALUE_MAPPING,
        association_table_seed=ASSOCIATION_TABLE_SEED,
        include_slices=True,
    )


def data_config(task_name: str, seed: int):
    spec = TASK_SPECS[task_name]
    num_stages = spec["num_stages"]
    return ContinualDataConfig(
        train_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=spec["train_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        test_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=spec["test_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def model_by_key(model_key: str):
    if model_key == "attention":
        return transformer_model(
            input_seq_len=INPUT_SEQ_LEN,
            vocab_size=VOCAB_SIZE,
        )
    if model_key == "base_rmt_nmem4":
        return base_rmt_model(
            input_seq_len=INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=4,
            vocab_size=VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem4"})
    if model_key == "fmrmt_stable":
        return fast_memory_rmt_model(
            input_seq_len=INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.005,
            rmt_slow_update_freq=4,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_stable_lr0p005_slow4"})
    raise ValueError(f"Unknown PERMUTED_AR_SMOKE_MODELS entry: {model_key}")


configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=model_by_key(model_key),
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "permuted_mapping",
                "smoke",
                task_name,
            ]
        ),
        max_epochs=TASK_SPECS[task_name]["max_epochs"],
        learning_rate=LEARNING_RATE,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        train_log_interval=0,
        continual_epoch_eval_interval=0,
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
        run_id=f"permuted-smoke-{task_name}-{model_key}-seed{seed}",
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for model_key in MODEL_KEYS
]


class CaptureLogger:
    instances = []

    def __init__(self, config):
        self.config = config
        self.config_dump = None
        self.history = []
        self.model_info = {}
        CaptureLogger.instances.append(self)

    def log_config(self, config):
        self.config_dump = config.model_dump()

    def log_model(self, model, config):
        test_configs = getattr(config.data, "test_stage_configs", [])
        max_seq_len = max(c.input_seq_len for c in test_configs)
        self.model_info = {
            "num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "state_size": model.state_size(sequence_length=max_seq_len),
        }

    def log(self, metrics):
        self.history.append(dict(metrics))

    def finish(self):
        pass


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


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _final_summary(history: list[dict[str, Any]]):
    candidates = [entry for entry in history if "continual/seen_avg_accuracy" in entry]
    if not candidates:
        raise RuntimeError("No continual summary metrics captured")
    return {key: _json_safe(value) for key, value in candidates[-1].items()}


def _task_and_model_from_run_id(run_id: str):
    body = run_id.removeprefix("permuted-smoke-").rsplit("-seed", 1)[0]
    for model_key in sorted(MODEL_KEYS, key=len, reverse=True):
        suffix = f"-{model_key}"
        if body.endswith(suffix):
            return body[: -len(suffix)], model_key
    raise ValueError(f"Cannot parse run_id: {run_id}")


def _model_settings(config: ContinualTrainConfig):
    kwargs = config.model.sequence_mixer.kwargs if config.model.sequence_mixer else {}
    return {
        key: kwargs[key]
        for key in [
            "n_mem",
            "rmt_fast_lr",
            "rmt_slow_update_freq",
            "reset_memory_each_batch",
            "reset_memory_each_epoch",
            "rmt_clip_memory_grad",
        ]
        if key in kwargs
    }


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
        "final_cumulative_wall_seconds": "continual/cumulative_wall_seconds",
    }
    grouped = {}
    for run in runs:
        key = f"{run['task_name']}::{run['model_key']}"
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
            "metrics": metrics,
        }
    return aggregates


def run_all(output_prefix: str = "class_incremental_ar_permuted_smoke"):
    import torch
    import zoology.train as train_module

    original_logger = train_module.WandbLogger
    original_tqdm = train_module.tqdm
    train_module.WandbLogger = CaptureLogger
    train_module.tqdm = QuietTqdm

    runs = []
    try:
        total = len(configs)
        for idx, config in enumerate(configs, start=1):
            CaptureLogger.instances.clear()
            task_name, model_key = _task_and_model_from_run_id(config.run_id)
            print(f"[{idx}/{total}] {config.run_id}")
            train_module.train_continual(config)
            logger = CaptureLogger.instances[-1]
            summary = _final_summary(logger.history)
            run = {
                "run_id": config.run_id,
                "task_name": task_name,
                "model_key": model_key,
                "model_name": config.model.name,
                "seed": config.seed,
                "model_settings": _model_settings(config),
                "model_info": logger.model_info,
                "summary": summary,
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
        train_module.WandbLogger = original_logger
        train_module.tqdm = original_tqdm

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_slug = "_".join(TASK_NAMES)
    output_path = Path("results") / f"{output_prefix}_{task_slug}_{created_at}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "experiment": "class-incremental AR with fixed per-stage random value permutations",
        "module": __name__,
        "setup": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tasks": TASK_NAMES,
            "models": MODEL_KEYS,
            "seeds": SEEDS,
            "vocab_size": VOCAB_SIZE,
            "associations_per_stage": ASSOCIATIONS_PER_STAGE,
            "num_query_associations": NUM_QUERY_ASSOCIATIONS,
            "input_seq_len": INPUT_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "segment_len": SEGMENT_LEN,
            "learning_rate": LEARNING_RATE,
            "value_mapping": VALUE_MAPPING,
            "association_table_seed": ASSOCIATION_TABLE_SEED,
            "stage_local_random_accuracy": 1.0 / ASSOCIATIONS_PER_STAGE,
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
