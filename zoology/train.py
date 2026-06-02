import argparse
import os
import random
import time
from datetime import datetime
from typing import List, Union
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from einops import rearrange

from zoology.data.utils import prepare_data, prepare_continuous_data
from zoology.config import ContinualTrainConfig, DataConfig, TrainConfig
from zoology.model import LanguageModel, ContinuousInputModel
from zoology.logger import WandbLogger
from zoology.utils import set_determinism
from zoology.metrics import compute_mse, compute_ce_with_embeddings


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        input_type: str = "discrete",
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.1,
        early_stopping_metric: str = None,
        early_stopping_threshold: float = None,
        train_log_interval: int = 1,
        loss_type: str = "ce",
        slice_keys: List[str] = [],
        slow_update_mode: str = "skip",
        device: Union[str, int] = "cuda",
        logger: WandbLogger = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.input_type = input_type
        self.logger = logger

        self.device = device
        self.max_epochs = max_epochs
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.slice_keys = slice_keys
        self.train_log_interval = train_log_interval
        self.loss_type = loss_type
        if slow_update_mode not in {"skip", "accumulate"}:
            raise ValueError("slow_update_mode must be 'skip' or 'accumulate'")
        self.slow_update_mode = slow_update_mode
        self.log_context = {}
        self.global_train_batch = 0
        self.global_optimizer_step = 0

    def _modules_with_hook(self, hook_name: str):
        return [
            module
            for module in self.model.modules()
            if hasattr(module, hook_name)
        ]

    def _call_model_hook(self, hook_name: str, **kwargs):
        for module in self._modules_with_hook(hook_name):
            getattr(module, hook_name)(**kwargs)

    def _fast_memory_parameter_ids(self):
        parameter_ids = set()
        for module in self._modules_with_hook("fast_memory_parameters"):
            parameter_ids.update(id(p) for p in module.fast_memory_parameters())
        return parameter_ids

    def _optimizer_parameters(self):
        fast_memory_parameter_ids = self._fast_memory_parameter_ids()
        return [
            p
            for p in self.model.parameters()
            if p.requires_grad and id(p) not in fast_memory_parameter_ids
        ]

    def _scale_fast_memory_grads(self, scale: float):
        if scale == 1.0:
            return
        for module in self._modules_with_hook("fast_memory_parameters"):
            for parameter in module.fast_memory_parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)

    def _slow_update_freq(self):
        freqs = [
            getattr(module, "rmt_slow_update_freq", 1)
            for module in self._modules_with_hook("update_fast_memory")
        ]
        return max(freqs) if freqs else 1

    def _optimizer_steps_per_epoch(self, dataloader: DataLoader = None):
        dataloader = dataloader or self.train_dataloader
        slow_update_freq = self._slow_update_freq()
        if self.slow_update_mode == "accumulate":
            return int(np.ceil(len(dataloader) / slow_update_freq))
        return len(dataloader) // slow_update_freq

    def _collect_model_diagnostics(self):
        diagnostics = {}
        aggregate = {}
        for module_idx, module in enumerate(self._modules_with_hook("get_diagnostics")):
            module_diagnostics = module.get_diagnostics()
            for key, value in module_diagnostics.items():
                diagnostics[f"train/{module.__class__.__name__}.{module_idx}/{key}"] = value
                aggregate.setdefault(key, []).append(value)

        for key, values in aggregate.items():
            diagnostics[f"train/{key}"] = float(np.mean(values))
        return diagnostics

    def on_before_optimizer_step(self):
        return None

    def on_after_optimizer_step(self, optimizer_step_context):
        return None

    def compute_loss(self, inputs, targets):
        if self.input_type == "continuous":
            
            all_embeddings = self.model.backbone.embeddings.word_embeddings.weight
            vocab_size = all_embeddings.shape[0]
            embed_dim = all_embeddings.shape[1]
            value_embeddings = all_embeddings[vocab_size // 2:]  # all values as candidates
            
            outputs = self.model(inputs, return_embeddings=True)
            num_kv_pairs = targets.shape[1]
            outputs = outputs[:, -num_kv_pairs:]
            
            outputs_flat = outputs.reshape(-1, embed_dim)
            targets_flat = targets.reshape(-1)
            
            if self.loss_type == "mse":
                target_embeds = value_embeddings[targets_flat]
                loss, _ = compute_mse(outputs_flat, target_embeds)
            else:  # ce or ce_embed
                loss, _ = compute_ce_with_embeddings(
                    outputs_flat, targets_flat, value_embeddings
                )
            
            logits = outputs_flat @ value_embeddings.T
            preds = (logits).argmax(dim=-1).view(targets.shape)
            return loss, preds
        
        else: # discrete
            if self.loss_type == "ce":
                logits = self.model(inputs, return_embeddings=False)
                loss = self.loss_fn(
                    rearrange(logits, "... c -> (...) c"), 
                    targets.flatten()
                )
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "mse":
                embeddings = self.model(inputs, return_embeddings=True)
                target_embeds = self.model.backbone.embeddings.word_embeddings(targets)
                mask = (targets != -100).unsqueeze(-1)
                loss, _ = compute_mse(
                    embeddings[mask.expand_as(embeddings)].view(-1, embeddings.size(-1)),
                    target_embeds[mask.expand_as(target_embeds)].view(-1, target_embeds.size(-1)),
                )
                logits = embeddings @ self.model.backbone.embeddings.word_embeddings.weight.T
                preds = logits.argmax(dim=-1)
                return loss, preds
            
            elif self.loss_type == "ce_embed":
                embeddings = self.model(inputs, return_embeddings=True)
                value_embeddings = self.model.backbone.embeddings.word_embeddings.weight
                flat_embeds = rearrange(embeddings, "b s d -> (b s) d")
                flat_targets = targets.flatten()
                mask = flat_targets != -100
                loss, _ = compute_ce_with_embeddings(
                    flat_embeds[mask], flat_targets[mask], value_embeddings,
                )
                logits = embeddings @ value_embeddings.T
                preds = logits.argmax(dim=-1)
                return loss, preds

    def train_epoch(
        self,
        epoch_idx: int,
        desc: str = None,
        log_context: dict = None,
    ):
        self.model.train()
        self._call_model_hook("on_epoch_start", epoch_idx=epoch_idx)
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=desc or f"Train Epoch {epoch_idx + 1}/{self.max_epochs}",
        )

        log_context = log_context or {}
        loss_values = []
        slow_update_freq = self._slow_update_freq()
        accumulate_slow_grads = (
            self.slow_update_mode == "accumulate" and slow_update_freq > 1
        )
        tail_batches = len(self.train_dataloader) % slow_update_freq
        if accumulate_slow_grads:
            self.optimizer.zero_grad()

        for batch_idx, (inputs, targets, slices) in enumerate(iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self._call_model_hook(
                "on_batch_start",
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
            if not accumulate_slow_grads:
                self.optimizer.zero_grad()

            loss, preds = self.compute_loss(inputs, targets)

            # Auxiliary losses (discrete mode only)
            if self.input_type == "discrete":
                auxiliary_loss = []
                def get_auxiliary_loss(module):
                    if hasattr(module, "get_auxiliary_loss"):
                        auxiliary_loss.append(module.get_auxiliary_loss())
                self.model.apply(get_auxiliary_loss)
                if auxiliary_loss:
                    loss = loss + sum(auxiliary_loss)

            if accumulate_slow_grads:
                is_tail_accumulation = (
                    tail_batches != 0
                    and batch_idx >= len(self.train_dataloader) - tail_batches
                )
                accumulation_divisor = (
                    tail_batches if is_tail_accumulation else slow_update_freq
                )
                (loss / accumulation_divisor).backward()
                # Keep the fast-memory step at per-batch scale while averaging
                # only the slow-parameter gradients over the accumulation window.
                self._scale_fast_memory_grads(accumulation_divisor)
                do_slow_step = (
                    (batch_idx + 1) % slow_update_freq == 0
                    or batch_idx + 1 == len(self.train_dataloader)
                )
            else:
                accumulation_divisor = 1
                loss.backward()
                do_slow_step = (batch_idx + 1) % slow_update_freq == 0

            self._call_model_hook("update_fast_memory")
            if do_slow_step:
                optimizer_step_context = self.on_before_optimizer_step()
                self.optimizer.step()
                self.on_after_optimizer_step(optimizer_step_context)
                self.global_optimizer_step += 1
                if getattr(self, "scheduler_steps_on_optimizer_step", False):
                    self.scheduler.step()
                if accumulate_slow_grads:
                    self.optimizer.zero_grad()
            self.global_train_batch += 1

            loss_value = loss.item()
            loss_values.append(loss_value)
            iterator.set_postfix({"loss": loss_value})

            should_log_batch = (
                self.train_log_interval > 0
                and (
                    self.global_train_batch == 1
                    or self.global_train_batch % self.train_log_interval == 0
                    or batch_idx + 1 == len(self.train_dataloader)
                )
            )
            if should_log_batch:
                diagnostics = self._collect_model_diagnostics()
                self.logger.log({
                    "train/loss": loss_value,
                    "train/lr": self.current_learning_rate(),
                    "train/slow_update_mode_is_accumulate": float(self.slow_update_mode == "accumulate"),
                    "train/slow_update_freq": slow_update_freq,
                    "train/accumulation_divisor": accumulation_divisor,
                    "train/global_batch": self.global_train_batch,
                    "train/global_optimizer_step": self.global_optimizer_step,
                    "train/batch_in_epoch": batch_idx + 1,
                    "epoch": epoch_idx,
                    **self.log_context,
                    **log_context,
                    **diagnostics,
                })

        return {
            "train/epoch_loss": float(np.mean(loss_values)),
            "train/epoch_batches": len(loss_values),
            "train/lr": self.current_learning_rate(),
            "train/slow_update_mode_is_accumulate": float(self.slow_update_mode == "accumulate"),
            "train/slow_update_freq": slow_update_freq,
            "train/global_batch": self.global_train_batch,
            "train/global_optimizer_step": self.global_optimizer_step,
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        metric_prefix: str = "valid",
        desc: str = None,
        log: bool = True,
        log_context: dict = None,
    ):
        self.model.eval()
        test_loss = 0
        results = []

        with torch.no_grad(), tqdm(
            total=len(dataloader),
            desc=desc or f"Valid Epoch {epoch_idx + 1}/{self.max_epochs}",
            postfix={"loss": "-", "acc": "-"},
        ) as iterator:
            for inputs, targets, slices in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss, preds = self.compute_loss(inputs, targets)
                test_loss += loss / len(dataloader)
                results.extend(compute_metrics(preds.cpu(), targets.cpu(), slices))
                iterator.update(1)

            results = pd.DataFrame(results)
            test_accuracy = results["accuracy"].mean()

            # logging and printing
            metrics = {
                f"{metric_prefix}/loss": test_loss.item(),
                f"{metric_prefix}/accuracy": test_accuracy.item(),
            }

            # compute metrics for slices
            for key in self.slice_keys:
                if key in results:
                    acc_by_slice = results.groupby(key)["accuracy"].mean()
                    for value, accuracy in acc_by_slice.items():
                        metrics[f"{metric_prefix}/{key}/accuracy-{value}"] = accuracy

            iterator.set_postfix(metrics)
            if log:
                self.logger.log({
                    "epoch": epoch_idx,
                    **self.log_context,
                    **(log_context or {}),
                    **metrics,
                })
        return metrics

    def test(self, epoch_idx: int):
        return self.evaluate(
            self.test_dataloader,
            epoch_idx=epoch_idx,
            metric_prefix="valid",
            log=True,
        )

    def _set_optimizer_lr(self, learning_rate: float):
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def current_learning_rate(self) -> float:
        if not hasattr(self, "optimizer") or not self.optimizer.param_groups:
            return float(self.learning_rate)
        return float(self.optimizer.param_groups[0]["lr"])

    def reset_scheduler(
        self,
        scheduler_epochs: int = None,
        scheduler_mode: str = None,
        scheduler_steps_per_epoch: int = None,
        reset_lr: bool = True,
    ):
        scheduler_mode = scheduler_mode or getattr(
            self, "scheduler_mode", "global_cosine"
        )
        self.scheduler_mode = scheduler_mode
        self.scheduler_steps_on_optimizer_step = False
        if reset_lr:
            self._set_optimizer_lr(self.learning_rate)

        if scheduler_mode in {"global_cosine", "stage_cosine"}:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_epochs or self.max_epochs,
                eta_min=0.0,
            )
        elif scheduler_mode == "constant":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda _: 1.0,
            )
        elif scheduler_mode == "stage_onecycle":
            steps_per_epoch = scheduler_steps_per_epoch
            if steps_per_epoch is None:
                steps_per_epoch = max(1, self._optimizer_steps_per_epoch())
            scheduler_epochs = scheduler_epochs or self.max_epochs
            # PyTorch's linear OneCycleLR reaches the final LR one call before
            # its nominal total_steps and can go negative on the last exact
            # step. Add one internal scheduler slot so the last optimizer step
            # lands on the configured final LR.
            total_steps = max(1, steps_per_epoch) * scheduler_epochs + 1
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                anneal_strategy="linear",
                total_steps=total_steps,
            )
            self.scheduler_steps_on_optimizer_step = True
        else:
            raise ValueError(
                "scheduler_mode must be one of "
                "'global_cosine', 'stage_cosine', 'stage_onecycle', "
                "or 'constant'"
            )

    def initialize_training(
        self,
        scheduler_epochs: int = None,
        scheduler_mode: str = "global_cosine",
        scheduler_steps_per_epoch: int = None,
    ):
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self._optimizer_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.reset_scheduler(
            scheduler_epochs=scheduler_epochs or self.max_epochs,
            scheduler_mode=scheduler_mode,
            scheduler_steps_per_epoch=scheduler_steps_per_epoch,
            reset_lr=False,
        )

    def fit(self):
        self.initialize_training()
        for epoch_idx in range(self.max_epochs):
            train_metrics = self.train_epoch(epoch_idx)
            self.logger.log({"epoch": epoch_idx, **train_metrics})
            metrics = self.test(epoch_idx)

            # early stopping
            if (self.early_stopping_metric is not None) and metrics[
                self.early_stopping_metric
            ] > self.early_stopping_threshold:
                print(
                    f"Early stopping triggered at epoch {epoch_idx} with "
                    f"{self.early_stopping_metric} {metrics[self.early_stopping_metric]} > {self.early_stopping_threshold}"
                )
                break

            if not getattr(self, "scheduler_steps_on_optimizer_step", False):
                self.scheduler.step()

    def on_continual_stage_end(self, stage_idx: int, train_dataloader: DataLoader):
        return {}


def compute_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    slices: List[dict],
    ignore_index: int = -100,
):
    results = []
    for pred, target, slc in zip(preds, targets, slices):
        results.append(
            {
                "accuracy": (pred == target)[target != ignore_index].to(float).mean().item(),
                **slc
            }
        )
    return results


def train(config: TrainConfig):
    set_determinism(config.seed)
    
    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    if config.input_type == "continuous":
        model = ContinuousInputModel(config.model)
        train_dataloader, test_dataloader = prepare_continuous_data(
            config.data,
            embeddings=model.backbone.embeddings.word_embeddings.weight.detach(),
        )
    else:
        model = LanguageModel(config.model)
        train_dataloader, test_dataloader = prepare_data(config.data)

    logger.log_model(model, config=config)

    task = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )
    task.fit()
    logger.finish()


def _build_single_stage_dataloaders(config: ContinualTrainConfig, stage_idx: int):
    data_kwargs = {
        "train_configs": [config.data.train_stage_configs[stage_idx]],
        "test_configs": [config.data.test_stage_configs[stage_idx]],
        "batch_size": config.data.batch_size,
        "seed": config.data.seed + stage_idx,
        "force_cache": config.data.force_cache,
    }
    if config.data.cache_dir is not None:
        data_kwargs["cache_dir"] = config.data.cache_dir

    stage_data = DataConfig(
        **data_kwargs,
    )
    return tuple(prepare_data(stage_data))


def _summarize_continual_eval(
    stage_idx: int,
    stage_metrics: dict,
    best_stage_accuracy: dict,
    learning_stage_accuracy: dict = None,
    pre_learning_stage_accuracy: dict = None,
    pretrain_stage_accuracy: dict = None,
    stage_random_accuracy: dict = None,
):
    metrics = {"continual/stage": stage_idx}
    learning_stage_accuracy = learning_stage_accuracy or {}
    pre_learning_stage_accuracy = pre_learning_stage_accuracy or {}
    pretrain_stage_accuracy = pretrain_stage_accuracy or {}
    stage_random_accuracy = stage_random_accuracy or {}
    accuracies = []
    forgetting = []
    forgetting_from_learning = []
    old_stage_accuracies = []
    old_stage_forgetting = []
    old_stage_forgetting_from_learning = []
    old_stage_bwt = []

    for eval_stage_idx, eval_metrics in sorted(stage_metrics.items()):
        acc_key = f"continual/stage_{eval_stage_idx}/accuracy"
        loss_key = f"continual/stage_{eval_stage_idx}/loss"
        accuracy = eval_metrics[acc_key]

        previous_best = best_stage_accuracy.get(eval_stage_idx, accuracy)
        stage_forgetting = max(0.0, previous_best - accuracy)
        best_stage_accuracy[eval_stage_idx] = max(previous_best, accuracy)

        accuracies.append(accuracy)
        forgetting.append(stage_forgetting)
        if eval_stage_idx < stage_idx:
            old_stage_accuracies.append(accuracy)
            old_stage_forgetting.append(stage_forgetting)
        metrics[acc_key] = accuracy
        metrics[loss_key] = eval_metrics[loss_key]
        metrics[f"continual/stage_{eval_stage_idx}/forgetting"] = stage_forgetting

        if eval_stage_idx in learning_stage_accuracy:
            learning_accuracy = learning_stage_accuracy[eval_stage_idx]
            bwt = accuracy - learning_accuracy
            learning_forgetting = max(0.0, learning_accuracy - accuracy)
            metrics[f"continual/stage_{eval_stage_idx}/learning_accuracy"] = learning_accuracy
            metrics[f"continual/stage_{eval_stage_idx}/bwt"] = bwt
            metrics[
                f"continual/stage_{eval_stage_idx}/forgetting_from_learning"
            ] = learning_forgetting
            forgetting_from_learning.append(learning_forgetting)
            if eval_stage_idx < stage_idx:
                old_stage_bwt.append(bwt)
                old_stage_forgetting_from_learning.append(learning_forgetting)

        if eval_stage_idx in pre_learning_stage_accuracy:
            pre_learning_accuracy = pre_learning_stage_accuracy[eval_stage_idx]
            metrics[
                f"continual/stage_{eval_stage_idx}/pre_learning_accuracy"
            ] = pre_learning_accuracy

            if eval_stage_idx in pretrain_stage_accuracy:
                metrics[
                    f"continual/stage_{eval_stage_idx}/fwt_from_initial"
                ] = pre_learning_accuracy - pretrain_stage_accuracy[eval_stage_idx]

            if eval_stage_idx in stage_random_accuracy:
                metrics[
                    f"continual/stage_{eval_stage_idx}/fwt_from_random"
                ] = pre_learning_accuracy - stage_random_accuracy[eval_stage_idx]

    metrics["continual/seen_avg_accuracy"] = float(np.mean(accuracies))
    metrics["continual/avg_forgetting"] = float(np.mean(forgetting))
    metrics["continual/avg_forgetting_from_learning"] = (
        float(np.mean(forgetting_from_learning))
        if forgetting_from_learning
        else 0.0
    )
    metrics["continual/avg_bwt"] = (
        float(np.mean(old_stage_bwt))
        if old_stage_bwt
        else 0.0
    )
    metrics["continual/old_stage_avg_accuracy"] = (
        float(np.mean(old_stage_accuracies))
        if old_stage_accuracies
        else 0.0
    )
    metrics["continual/old_stage_avg_forgetting"] = (
        float(np.mean(old_stage_forgetting))
        if old_stage_forgetting
        else 0.0
    )
    metrics["continual/old_stage_avg_forgetting_from_learning"] = (
        float(np.mean(old_stage_forgetting_from_learning))
        if old_stage_forgetting_from_learning
        else 0.0
    )
    metrics["continual/old_stage_avg_bwt"] = (
        float(np.mean(old_stage_bwt))
        if old_stage_bwt
        else 0.0
    )

    current_stage_accuracy = stage_metrics[stage_idx][
        f"continual/stage_{stage_idx}/accuracy"
    ]
    current_stage_loss = stage_metrics[stage_idx][
        f"continual/stage_{stage_idx}/loss"
    ]
    metrics["continual/current_stage_accuracy"] = current_stage_accuracy
    metrics["continual/current_stage_loss"] = current_stage_loss
    metrics["continual/plasticity"] = current_stage_accuracy
    metrics[f"continual/stage_{stage_idx}/learning_accuracy"] = current_stage_accuracy

    learned_accuracies = [
        learning_stage_accuracy[learned_stage_idx]
        for learned_stage_idx in range(stage_idx + 1)
        if learned_stage_idx in learning_stage_accuracy
    ]
    if learned_accuracies:
        stage_0_learning_accuracy = learning_stage_accuracy.get(
            0,
            learned_accuracies[0],
        )
        metrics["continual/avg_learning_accuracy"] = float(
            np.mean(learned_accuracies)
        )
        metrics["continual/plasticity_drop_from_stage_0"] = (
            stage_0_learning_accuracy - current_stage_accuracy
        )
        metrics["continual/avg_plasticity_drop_from_stage_0"] = float(
            np.mean([
                stage_0_learning_accuracy - accuracy
                for accuracy in learned_accuracies
            ])
        )

    learned_fwt_from_random = [
        metrics[f"continual/stage_{learned_stage_idx}/fwt_from_random"]
        for learned_stage_idx in range(stage_idx + 1)
        if f"continual/stage_{learned_stage_idx}/fwt_from_random" in metrics
    ]
    if learned_fwt_from_random:
        metrics["continual/avg_fwt_from_random"] = float(
            np.mean(learned_fwt_from_random)
        )

    learned_fwt_from_initial = [
        metrics[f"continual/stage_{learned_stage_idx}/fwt_from_initial"]
        for learned_stage_idx in range(stage_idx + 1)
        if f"continual/stage_{learned_stage_idx}/fwt_from_initial" in metrics
    ]
    if learned_fwt_from_initial:
        metrics["continual/avg_fwt_from_initial"] = float(
            np.mean(learned_fwt_from_initial)
        )
    return metrics


def _normalized_stage_auc(values):
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    array = np.asarray(values, dtype=float)
    area = 0.5 * array[0] + array[1:-1].sum() + 0.5 * array[-1]
    return float(area / (len(array) - 1))


def _stage_random_accuracy(config: ContinualTrainConfig, stage_idx: int):
    stage_config = config.data.test_stage_configs[stage_idx]
    random_accuracy = getattr(stage_config, "random_accuracy", None)
    if callable(random_accuracy):
        return float(random_accuracy())
    return None


def _train_log_interval(config: TrainConfig):
    value = os.getenv("ZOOLOGY_TRAIN_LOG_INTERVAL")
    if value is None:
        return config.train_log_interval
    return int(value)


def _sync_device_for_timing(device):
    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()


def _dataloader_num_examples(dataloader: DataLoader):
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def _sequence_length_from_config(stage_config):
    return getattr(stage_config, "input_seq_len", None)


def _throughput_metrics(
    prefix: str,
    wall_seconds: float,
    examples: int = None,
    sequence_length: int = None,
):
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


def train_continual(config: ContinualTrainConfig):
    set_determinism(config.seed)

    logger = WandbLogger(config)
    logger.log_config(config)
    config.print()

    if config.input_type != "discrete":
        raise ValueError("Continual MQAR currently supports discrete input only.")

    if len(config.data.train_stage_configs) != len(config.data.test_stage_configs):
        raise ValueError("train_stage_configs and test_stage_configs must have the same length.")

    model = LanguageModel(config.model)
    logger.log_model(model, config=config)

    stage_dataloaders = [
        _build_single_stage_dataloaders(config, stage_idx)
        for stage_idx in range(len(config.data.train_stage_configs))
    ]

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
    seen_avg_accuracy_curve = []
    old_stage_avg_accuracy_curve = []
    old_stage_avg_forgetting_curve = []
    old_stage_avg_forgetting_from_learning_curve = []
    old_stage_avg_bwt_curve = []

    should_stop = False
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
            pre_learning_accuracy = pre_learning_metrics[
                f"continual/pre_learning/stage_{stage_idx}/accuracy"
            ]
            pre_learning_stage_accuracy[stage_idx] = pre_learning_accuracy

        current_test_dataloader = stage_dataloaders[stage_idx][1]
        epoch_eval_interval = config.continual_epoch_eval_interval
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
                    f"Train Continual Stage {stage_idx + 1}/{len(stage_dataloaders)} "
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
                epoch_eval_interval > 0
                and (
                    (local_epoch_idx + 1) % epoch_eval_interval == 0
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
                        f"Valid Continual Stage {stage_idx + 1}/{len(stage_dataloaders)} "
                        f"Epoch {local_epoch_idx + 1}/{config.max_epochs}"
                    ),
                    log=False,
                )
                _sync_device_for_timing(device)
                current_epoch_eval_wall = time.perf_counter() - epoch_eval_wall_start
                epoch_eval_wall_seconds += current_epoch_eval_wall

                current_epoch_accuracy = current_epoch_metrics[
                    "continual/current_stage_epoch/accuracy"
                ]
                current_epoch_loss = current_epoch_metrics[
                    "continual/current_stage_epoch/loss"
                ]
                logger.log({
                    "epoch": global_epoch_idx,
                    "continual/stage": stage_idx,
                    **epoch_log_context,
                    "continual/current_stage_epoch_index": local_epoch_idx,
                    "continual/current_stage_epoch_number": local_epoch_idx + 1,
                    "continual/current_stage_epoch_train_wall_seconds": epoch_train_wall,
                    "continual/current_stage_epoch_eval_wall_seconds": current_epoch_eval_wall,
                    f"continual/stage_{stage_idx}/epoch_accuracy": current_epoch_accuracy,
                    f"continual/stage_{stage_idx}/epoch_loss": current_epoch_loss,
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
                desc=f"Valid Continual Stage {eval_stage_idx}/{stage_idx}",
                log=False,
            )
            stage_metrics[eval_stage_idx] = metrics
        _sync_device_for_timing(device)
        seen_eval_wall_seconds = time.perf_counter() - seen_eval_wall_start

        learning_stage_accuracy[stage_idx] = stage_metrics[stage_idx][
            f"continual/stage_{stage_idx}/accuracy"
        ]

        metrics = _summarize_continual_eval(
            stage_idx=stage_idx,
            stage_metrics=stage_metrics,
            best_stage_accuracy=best_stage_accuracy,
            learning_stage_accuracy=learning_stage_accuracy,
            pre_learning_stage_accuracy=pre_learning_stage_accuracy,
            pretrain_stage_accuracy=pretrain_stage_accuracy,
            stage_random_accuracy=stage_random_accuracy,
        )
        seen_avg_accuracy_curve.append(metrics["continual/seen_avg_accuracy"])
        metrics["continual/seen_avg_accuracy_stage_auc"] = _normalized_stage_auc(
            seen_avg_accuracy_curve
        )
        if stage_idx > 0:
            old_stage_avg_accuracy_curve.append(
                metrics["continual/old_stage_avg_accuracy"]
            )
            old_stage_avg_forgetting_curve.append(
                metrics["continual/old_stage_avg_forgetting"]
            )
            old_stage_avg_forgetting_from_learning_curve.append(
                metrics["continual/old_stage_avg_forgetting_from_learning"]
            )
            old_stage_avg_bwt_curve.append(metrics["continual/old_stage_avg_bwt"])
        metrics["continual/old_stage_avg_accuracy_stage_auc"] = _normalized_stage_auc(
            old_stage_avg_accuracy_curve
        )
        metrics["continual/old_stage_avg_forgetting_stage_auc"] = _normalized_stage_auc(
            old_stage_avg_forgetting_curve
        )
        metrics[
            "continual/old_stage_avg_forgetting_from_learning_stage_auc"
        ] = _normalized_stage_auc(old_stage_avg_forgetting_from_learning_curve)
        metrics["continual/old_stage_avg_bwt_stage_auc"] = _normalized_stage_auc(
            old_stage_avg_bwt_curve
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
            trainer._optimizer_steps_per_epoch(train_dataloader)
            * config.max_epochs
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
        }
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

        stage_end_hook_metrics = trainer.on_continual_stage_end(
            stage_idx=stage_idx,
            train_dataloader=train_dataloader,
        )
        if stage_end_hook_metrics:
            logger.log({
                "epoch": stage_end_epoch,
                "continual/global_epoch": stage_end_epoch,
                "continual/local_epoch": config.max_epochs - 1,
                "continual/stage_epoch": config.max_epochs,
                "continual/stage": stage_idx,
                **stage_end_hook_metrics,
            })

        if config.evaluate_future_stages and stage_idx + 1 < len(stage_dataloaders):
            next_stage_idx = stage_idx + 1
            test_dataloader = stage_dataloaders[next_stage_idx][1]
            next_pre_learning_metrics = trainer.evaluate(
                test_dataloader,
                epoch_idx=(stage_idx + 1) * config.max_epochs - 1,
                metric_prefix=f"continual/pre_learning/stage_{next_stage_idx}",
                desc=f"Pre-Learning Continual Stage {next_stage_idx}",
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

        if (
            config.early_stopping_metric is not None
            and metrics.get(config.early_stopping_metric, -float("inf"))
            > config.early_stopping_threshold
        ):
            print(
                f"Early stopping triggered at continual stage {stage_idx} with "
                f"{config.early_stopping_metric} {metrics[config.early_stopping_metric]} > "
                f"{config.early_stopping_threshold}"
            )
            should_stop = True

        if should_stop:
            break

    logger.finish()


if __name__ == "__main__":
    config = TrainConfig.from_cli()
    train(config)
