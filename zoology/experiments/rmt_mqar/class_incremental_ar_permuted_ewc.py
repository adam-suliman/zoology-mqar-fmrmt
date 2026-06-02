"""Online EWC baseline for formal permuted class-incremental AR.

This runner keeps the AR stream non-replay: at stage t, the model trains only
on current-stage examples. After each stage it estimates a diagonal Fisher on
that stage's train loader and adds the online EWC penalty during later stages.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch

from zoology.config import ContinualTrainConfig, LoggerConfig
from zoology.experiments.rmt_mqar import class_incremental_ar_permuted_smoke as base
from zoology.experiments.rmt_mqar._common import transformer_model


TASK_SPECS = {
    "ewc3_permuted_smoke": {
        "num_stages": 3,
        "train_examples": 512,
        "test_examples": 256,
        "max_epochs": 4,
    },
    "formal20_permuted": {
        "num_stages": 20,
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
    },
}


def _csv_env(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


SEEDS = _csv_env("PERMUTED_AR_EWC_SEEDS", [123], int)
TASK_NAMES = _csv_env("PERMUTED_AR_EWC_TASKS", ["ewc3_permuted_smoke"], str)
EWC_LAMBDAS = _csv_env("PERMUTED_AR_EWC_LAMBDAS", [0.0, 10.0, 100.0], float)
EWC_DECAY = float(os.getenv("PERMUTED_AR_EWC_DECAY", "1.0"))
EWC_FISHER_MAX_BATCHES = int(os.getenv("PERMUTED_AR_EWC_FISHER_MAX_BATCHES", "0"))
EPOCH_EVAL_INTERVAL = int(os.getenv("PERMUTED_AR_EWC_EPOCH_EVAL_INTERVAL", "1"))
LR_SCHEDULER_MODE = os.getenv("PERMUTED_AR_EWC_LR_SCHEDULER_MODE", "stage_onecycle")
SLOW_UPDATE_MODE = os.getenv("PERMUTED_AR_EWC_SLOW_UPDATE_MODE", "accumulate")
if SLOW_UPDATE_MODE not in {"skip", "accumulate"}:
    raise ValueError("PERMUTED_AR_EWC_SLOW_UPDATE_MODE must be 'skip' or 'accumulate'")

base.TASK_SPECS.update(TASK_SPECS)
UNKNOWN_TASKS = sorted(set(TASK_NAMES) - set(base.TASK_SPECS))
if UNKNOWN_TASKS:
    raise ValueError(f"Unknown PERMUTED_AR_EWC_TASKS: {UNKNOWN_TASKS}")


CURRENT_EWC_SETTINGS: dict[str, float | int | str] = {}


def ewc_model_key(ewc_lambda: float) -> str:
    if ewc_lambda == 0.0:
        return "attention_no_ewc"
    return f"attention_online_ewc_lam{_float_slug(ewc_lambda)}"


def model_by_ewc_lambda(ewc_lambda: float):
    return transformer_model(
        input_seq_len=base.INPUT_SEQ_LEN,
        vocab_size=base.VOCAB_SIZE,
    ).model_copy(update={"name": ewc_model_key(ewc_lambda)})


def build_configs():
    run_specs = []
    for task_name in TASK_NAMES:
        task_spec = base.TASK_SPECS[task_name]
        for seed in SEEDS:
            for ewc_lambda in EWC_LAMBDAS:
                model_key = ewc_model_key(ewc_lambda)
                config = ContinualTrainConfig(
                    data=base.data_config(task_name=task_name, seed=seed),
                    model=model_by_ewc_lambda(ewc_lambda),
                    logger=LoggerConfig(
                        tags=[
                            "rmt_mqar",
                            "class_incremental_ar",
                            "permuted_mapping",
                            "online_ewc",
                            task_name,
                        ]
                    ),
                    max_epochs=task_spec["max_epochs"],
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
                    run_id=f"permuted-ewc-{task_name}-{model_key}-seed{seed}",
                )
                run_specs.append(
                    {
                        "config": config,
                        "task_name": task_name,
                        "model_key": model_key,
                        "ewc_lambda": ewc_lambda,
                    }
                )
    return run_specs


def _named_trainable_parameters(trainer):
    fast_memory_ids = trainer._fast_memory_parameter_ids()
    return [
        (name, parameter)
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and id(parameter) not in fast_memory_ids
    ]


class OnlineEWCTrainer:
    @classmethod
    def wrap(cls, base_trainer_cls):
        class _OnlineEWCTrainer(base_trainer_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.ewc_lambda = float(CURRENT_EWC_SETTINGS.get("ewc_lambda", 0.0))
                self.ewc_decay = float(CURRENT_EWC_SETTINGS.get("ewc_decay", 1.0))
                self.ewc_fisher_max_batches = int(
                    CURRENT_EWC_SETTINGS.get("ewc_fisher_max_batches", 0)
                )
                self.ewc_fisher: dict[str, torch.Tensor] | None = None
                self.ewc_reference: dict[str, torch.Tensor] | None = None
                self.last_ewc_penalty = 0.0

            def _ewc_penalty(self):
                if (
                    self.ewc_lambda == 0.0
                    or self.ewc_fisher is None
                    or self.ewc_reference is None
                ):
                    return None
                penalty = None
                for name, parameter in _named_trainable_parameters(self):
                    if name not in self.ewc_fisher:
                        continue
                    term = self.ewc_fisher[name] * (parameter - self.ewc_reference[name]).pow(2)
                    term = term.sum()
                    penalty = term if penalty is None else penalty + term
                if penalty is None:
                    return None
                return 0.5 * self.ewc_lambda * penalty

            def compute_loss(self, inputs, targets):
                loss, preds = super().compute_loss(inputs, targets)
                penalty = self._ewc_penalty()
                if penalty is not None:
                    self.last_ewc_penalty = float(penalty.detach().cpu())
                    loss = loss + penalty
                else:
                    self.last_ewc_penalty = 0.0
                return loss, preds

            def _estimate_stage_fisher(self, train_dataloader):
                parameters = _named_trainable_parameters(self)
                fisher = {
                    name: torch.zeros_like(parameter, device=parameter.device)
                    for name, parameter in parameters
                }
                was_training = self.model.training
                self.model.eval()
                self.optimizer.zero_grad(set_to_none=True)
                num_batches = 0
                loss_values = []
                max_batches = self.ewc_fisher_max_batches

                for batch_idx, (inputs, targets, _) in enumerate(train_dataloader):
                    if max_batches > 0 and batch_idx >= max_batches:
                        break
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, _ = super().compute_loss(inputs, targets)
                    loss.backward()
                    loss_values.append(float(loss.detach().cpu()))
                    for name, parameter in parameters:
                        if parameter.grad is not None:
                            fisher[name].add_(parameter.grad.detach().pow(2))
                    num_batches += 1

                if num_batches == 0:
                    raise RuntimeError("Cannot estimate EWC Fisher with zero batches")
                for name in fisher:
                    fisher[name].div_(num_batches)

                self.optimizer.zero_grad(set_to_none=True)
                self.model.train(was_training)
                return fisher, {
                    "continual/ewc_fisher_batches": num_batches,
                    "continual/ewc_fisher_loss": float(mean(loss_values)),
                }

            def on_continual_stage_end(self, stage_idx: int, train_dataloader):
                if self.ewc_lambda == 0.0:
                    return {
                        "continual/ewc_lambda": 0.0,
                        "continual/ewc_decay": self.ewc_decay,
                        "continual/ewc_active": 0.0,
                    }

                start = time.perf_counter()
                fisher, metrics = self._estimate_stage_fisher(train_dataloader)
                if self.ewc_fisher is None:
                    self.ewc_fisher = fisher
                else:
                    self.ewc_fisher = {
                        name: self.ewc_decay * self.ewc_fisher[name] + fisher[name]
                        for name in fisher
                    }
                self.ewc_reference = {
                    name: parameter.detach().clone()
                    for name, parameter in _named_trainable_parameters(self)
                }
                fisher_means = [
                    value.detach().abs().mean().item()
                    for value in self.ewc_fisher.values()
                ]
                metrics.update({
                    "continual/ewc_lambda": self.ewc_lambda,
                    "continual/ewc_decay": self.ewc_decay,
                    "continual/ewc_active": 1.0,
                    "continual/ewc_fisher_wall_seconds": time.perf_counter() - start,
                    "continual/ewc_fisher_mean_abs": float(mean(fisher_means)),
                    "continual/ewc_fisher_parameter_tensors": len(self.ewc_fisher),
                })
                return metrics

        return _OnlineEWCTrainer


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
            "seeds": sorted(run["seed"] for run in group),
            "task_spec": base.TASK_SPECS[first["task_name"]],
            "model_name": first["model_name"],
            "model_info": first["model_info"],
            "ewc_settings": first["ewc_settings"],
            "metrics": metrics,
        }
    return aggregates


def run_all(output_prefix: str = "class_incremental_ar_permuted_online_ewc"):
    import zoology.train as train_module

    run_specs = build_configs()
    original_logger = train_module.WandbLogger
    original_tqdm = train_module.tqdm
    original_trainer = train_module.Trainer
    train_module.WandbLogger = base.CaptureLogger
    train_module.tqdm = base.QuietTqdm
    train_module.Trainer = OnlineEWCTrainer.wrap(original_trainer)

    runs = []
    try:
        total = len(run_specs)
        for idx, run_spec in enumerate(run_specs, start=1):
            config = run_spec["config"]
            base.CaptureLogger.instances.clear()
            CURRENT_EWC_SETTINGS.clear()
            CURRENT_EWC_SETTINGS.update({
                "ewc_lambda": run_spec["ewc_lambda"],
                "ewc_decay": EWC_DECAY,
                "ewc_fisher_max_batches": EWC_FISHER_MAX_BATCHES,
            })
            print(f"[{idx}/{total}] {config.run_id}")
            train_module.train_continual(config)
            logger = base.CaptureLogger.instances[-1]
            summary = _final_summary(logger.history)
            task_spec = base.TASK_SPECS[run_spec["task_name"]]
            run = {
                "run_id": config.run_id,
                "task_name": run_spec["task_name"],
                "model_key": run_spec["model_key"],
                "model_name": config.model.name,
                "seed": config.seed,
                "model_info": logger.model_info,
                "lr_scheduler_mode": getattr(config, "lr_scheduler_mode", "global_cosine"),
                "slow_update_mode": getattr(config, "slow_update_mode", "skip"),
                "continual_epoch_eval_interval": getattr(config, "continual_epoch_eval_interval", 0),
                "ewc_settings": dict(CURRENT_EWC_SETTINGS),
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
        train_module.WandbLogger = original_logger
        train_module.tqdm = original_tqdm
        train_module.Trainer = original_trainer
        CURRENT_EWC_SETTINGS.clear()

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_slug = "_".join(TASK_NAMES)
    output_path = Path("results") / f"{output_prefix}_{task_slug}_{created_at}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "experiment": "online EWC baseline for permuted class-incremental AR",
        "module": __name__,
        "setup": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tasks": TASK_NAMES,
            "seeds": SEEDS,
            "ewc_lambdas": EWC_LAMBDAS,
            "ewc_decay": EWC_DECAY,
            "ewc_fisher_max_batches": EWC_FISHER_MAX_BATCHES,
            "vocab_size": base.VOCAB_SIZE,
            "associations_per_stage": base.ASSOCIATIONS_PER_STAGE,
            "num_query_associations": base.NUM_QUERY_ASSOCIATIONS,
            "input_seq_len": base.INPUT_SEQ_LEN,
            "batch_size": base.BATCH_SIZE,
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
        "task_specs": {name: base.TASK_SPECS[name] for name in TASK_NAMES},
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
