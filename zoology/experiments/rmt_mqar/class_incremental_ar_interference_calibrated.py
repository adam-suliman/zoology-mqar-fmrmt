"""Calibrated repeated-key latest-value interference experiments.

The original repeated-key latest-value task used 8 queried associations but only
4 repeated-key updates. A model can score near 0.5 by retrieving the original
values and ignoring updates, because half the queried keys are unchanged.

This module adds all-updated variants where every queried key is repeated with a
changed value. In those variants latest-value accuracy directly measures overwrite
learning. Use environment variables to select cheap screens or continual runs,
for example:

``INTERFERENCE_CALIBRATION_TASKS=one_stage_latest_all8_budget16``
``INTERFERENCE_CALIBRATION_MODELS=attention,base_rmt_nmem16,fmrmt_stable,fmrmt_plastic``
``INTERFERENCE_CALIBRATION_SEEDS=123,456,789``
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.interference_ar import InterferenceARConfig
from zoology.experiments.rmt_mqar._common import (
    base_rmt_model,
    fast_memory_rmt_model,
    transformer_model,
)


VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
INPUT_SEQ_LEN = 128
BATCH_SIZE = 64
SEGMENT_LEN = 64
LEARNING_RATE = 3e-3

TASK_SPECS = {
    "one_stage_latest_mixed4_low": {
        "num_stages": 1,
        "num_query_associations": 8,
        "num_interference_pairs": 4,
        "target_policy": "latest",
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "Original mixed updated/unchanged one-stage calibration.",
    },
    "one_stage_latest_all4_low": {
        "num_stages": 1,
        "num_query_associations": 4,
        "num_interference_pairs": 4,
        "target_policy": "latest",
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "Cheap all-updated one-stage overwrite check.",
    },
    "one_stage_latest_all8_low": {
        "num_stages": 1,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "All-updated one-stage check with the standard 8 labels/sequence.",
    },
    "one_stage_latest_all8_budget16": {
        "num_stages": 1,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
        "purpose": "Higher-budget all-updated one-stage solvability check.",
    },
    "one_stage_latest_fixed_all8_low": {
        "num_stages": 1,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "update_value_mode": "fixed_shift",
        "fixed_update_offset": 1,
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "Fixed-update all-updated one-stage overwrite check.",
    },
    "one_stage_latest_fixed_all8_budget16": {
        "num_stages": 1,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "update_value_mode": "fixed_shift",
        "fixed_update_offset": 1,
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
        "purpose": "Higher-budget fixed-update all-updated one-stage check.",
    },
    "continual_latest_fixed_all8_low": {
        "num_stages": 5,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "update_value_mode": "fixed_shift",
        "fixed_update_offset": 1,
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "Cheap 5-stage fixed-update all-updated latest-value continual run.",
    },
    "continual_latest_fixed_all8_budget16": {
        "num_stages": 5,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "update_value_mode": "fixed_shift",
        "fixed_update_offset": 1,
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
        "purpose": "Higher-budget 5-stage fixed-update all-updated latest-value continual run.",
    },
    "continual_latest_all8_low": {
        "num_stages": 5,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
        "purpose": "Cheap 5-stage all-updated latest-value continual run.",
    },
    "continual_latest_all8_budget16": {
        "num_stages": 5,
        "num_query_associations": 8,
        "num_interference_pairs": 8,
        "target_policy": "latest",
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
        "purpose": "Higher-budget 5-stage all-updated latest-value continual run.",
    },
}


def _csv_env(name: str, default: list[Any], cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


SEEDS = _csv_env("INTERFERENCE_CALIBRATION_SEEDS", [123], int)
TASK_NAMES = _csv_env(
    "INTERFERENCE_CALIBRATION_TASKS",
    ["one_stage_latest_mixed4_low", "one_stage_latest_all4_low", "one_stage_latest_all8_low"],
    str,
)
MODEL_KEYS = _csv_env(
    "INTERFERENCE_CALIBRATION_MODELS",
    ["attention", "base_rmt_nmem16", "fmrmt_stable", "fmrmt_plastic"],
    str,
)

UNKNOWN_TASKS = sorted(set(TASK_NAMES) - set(TASK_SPECS))
if UNKNOWN_TASKS:
    raise ValueError(f"Unknown interference calibration tasks: {UNKNOWN_TASKS}")


def _float_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def stage_config(
    spec: dict[str, Any],
    stage_idx: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
):
    return InterferenceARConfig(
        stage_idx=stage_idx,
        num_stages=spec["num_stages"],
        associations_per_stage=ASSOCIATIONS_PER_STAGE,
        num_query_associations=spec["num_query_associations"],
        num_interference_pairs=spec["num_interference_pairs"],
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        target_policy=spec["target_policy"],
        update_value_mode=spec.get("update_value_mode", "random"),
        fixed_update_offset=spec.get("fixed_update_offset", 1),
        include_slices=True,
    )


def data_config(task_name: str, seed: int):
    spec = TASK_SPECS[task_name]
    return ContinualDataConfig(
        train_stage_configs=[
            stage_config(
                spec=spec,
                stage_idx=stage_idx,
                num_examples=spec["train_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(spec["num_stages"])
        ],
        test_stage_configs=[
            stage_config(
                spec=spec,
                stage_idx=stage_idx,
                num_examples=spec["test_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(spec["num_stages"])
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def model_for_key(model_key: str):
    if model_key == "attention":
        return transformer_model(input_seq_len=INPUT_SEQ_LEN, vocab_size=VOCAB_SIZE)
    if model_key == "base_rmt_nmem4":
        return base_rmt_model(
            input_seq_len=INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=4,
            vocab_size=VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem4"})
    if model_key == "base_rmt_nmem16":
        return base_rmt_model(
            input_seq_len=INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=16,
            vocab_size=VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem16"})
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
    if model_key == "fmrmt_plastic":
        return fast_memory_rmt_model(
            input_seq_len=INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.01,
            rmt_slow_update_freq=1,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_plastic_lr0p01_slow1"})
    raise ValueError(f"Unknown model key: {model_key}")


configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=model_for_key(model_key),
        logger=LoggerConfig(
            tags=["rmt_mqar", "class_incremental_ar", "interference_calibrated", task_name]
        ),
        max_epochs=TASK_SPECS[task_name]["max_epochs"],
        learning_rate=LEARNING_RATE,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        slice_keys=[
            "stage_idx",
            "eval_mode",
            "associations_per_stage",
            "num_query_associations",
            "num_interference_pairs",
            "target_policy",
            "update_value_mode",
            "fixed_update_offset",
            "input_seq_len",
        ],
        seed=seed,
        run_id=f"interference-calibrated-{task_name}-{model_key}-seed{seed}",
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for model_key in MODEL_KEYS
]


class CaptureLogger:
    instances = []

    def __init__(self, config):
        self.config = config
        self.history = []
        self.model_info = {}
        CaptureLogger.instances.append(self)
        print("Capture logger active")

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
    final = candidates[-1]
    return {key: _json_safe(value) for key, value in final.items()}


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


def _condition(task_name: str):
    spec = dict(TASK_SPECS[task_name])
    spec.pop("purpose", None)
    spec["condition_name"] = task_name
    return spec


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
        "final_train_loss": "train/loss",
        "final_cumulative_wall_seconds": "continual/cumulative_wall_seconds",
        "final_cumulative_train_wall_seconds": "continual/cumulative_train_wall_seconds",
        "final_cumulative_seen_eval_wall_seconds": "continual/cumulative_seen_eval_wall_seconds",
        "mean_stage_wall_seconds": "continual/stage_wall_seconds",
        "mean_stage_train_wall_seconds": "continual/stage_train_wall_seconds",
        "mean_stage_seen_eval_wall_seconds": "continual/stage_seen_eval_wall_seconds",
        "mean_stage_train_tokens_per_second": "continual/stage_train_tokens_per_second",
        "mean_stage_seen_eval_tokens_per_second": "continual/stage_seen_eval_tokens_per_second",
    }
    grouped = {}
    for run in runs:
        key = f"{run['task_name']}::{run['model_name']}"
        grouped.setdefault(key, []).append(run)

    aggregates = {}
    for key, group in grouped.items():
        metrics = {}
        for out_name, source_name in metric_map.items():
            values = [
                run["summary"][source_name]
                for run in group
                if source_name in run["summary"] and isinstance(run["summary"][source_name], (int, float))
            ]
            if values:
                metrics[out_name] = _stat(values)
        first = group[0]
        aggregates[key] = {
            "num_runs": len(group),
            "condition": first["condition"],
            "model_info": first["model_info"],
            "model_settings": first["model_settings"],
            "metrics": metrics,
        }
    return aggregates


def run_all(output_prefix: str = "interference_ar_calibrated"):
    import torch
    import zoology.train as train_module

    original_logger = train_module.WandbLogger
    original_tqdm = train_module.tqdm
    train_module.WandbLogger = CaptureLogger
    if os.getenv("INTERFERENCE_CALIBRATION_QUIET", "1").lower() not in {"0", "false", "no", "off"}:
        train_module.tqdm = QuietTqdm
    runs = []
    try:
        total = len(configs)
        for idx, config in enumerate(configs, start=1):
            CaptureLogger.instances.clear()
            task_name = config.run_id.split("interference-calibrated-", 1)[1].rsplit("-seed", 1)[0]
            for model_key in MODEL_KEYS:
                suffix = f"-{model_key}"
                if task_name.endswith(suffix):
                    task_name = task_name[: -len(suffix)]
                    break
            print(f"[{idx}/{total}] {config.run_id}")
            train_module.train_continual(config)
            logger = CaptureLogger.instances[-1]
            summary = _final_summary(logger.history)
            run = {
                "run_id": config.run_id,
                "task_name": task_name,
                "seed": config.seed,
                "model_name": config.model.name,
                "model_settings": _model_settings(config),
                "model_info": logger.model_info,
                "config": getattr(logger, "config_dump", None),
                "condition": _condition(task_name),
                "summary": summary,
                "history": logger.history,
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

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_slug = "_".join(TASK_NAMES)
    if len(task_slug) > 80:
        task_slug = "multi"
    output_path = Path("results") / f"{output_prefix}_{task_slug}_formal_cl_{created_at}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "experiment": "calibrated repeated-key latest-value interference",
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
            "input_seq_len": INPUT_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "segment_len": SEGMENT_LEN,
            "learning_rate": LEARNING_RATE,
            "stage_local_random_accuracy": 1.0 / ASSOCIATIONS_PER_STAGE,
            "timing_notes": [
                "Wall-clock timings are measured by train_continual with CUDA synchronization around train and seen-eval blocks.",
                "All-updated variants set num_interference_pairs == num_query_associations, so latest-value accuracy directly measures overwrite learning.",
            ],
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
