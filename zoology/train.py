import argparse
import random
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
        loss_type: str = "ce",
        slice_keys: List[str] = [],
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
        self.loss_type = loss_type
        self.log_context = {}

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

    def _slow_update_freq(self):
        freqs = [
            getattr(module, "rmt_slow_update_freq", 1)
            for module in self._modules_with_hook("update_fast_memory")
        ]
        return max(freqs) if freqs else 1

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

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        self._call_model_hook("on_epoch_start", epoch_idx=epoch_idx)
        iterator = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Train Epoch {epoch_idx}/{self.max_epochs}",
        )

        slow_update_freq = self._slow_update_freq()
        for batch_idx, (inputs, targets, slices) in enumerate(iterator):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self._call_model_hook(
                "on_batch_start",
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
            )
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

            loss.backward()
            self._call_model_hook("update_fast_memory")
            if (batch_idx + 1) % slow_update_freq == 0:
                self.optimizer.step()

            diagnostics = self._collect_model_diagnostics()
            iterator.set_postfix({"loss": loss.item()})
            self.logger.log({
                "train/loss": loss.item(),
                "epoch": epoch_idx,
                **self.log_context,
                **diagnostics,
            })

    def evaluate(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        metric_prefix: str = "valid",
        desc: str = None,
        log: bool = True,
    ):
        self.model.eval()
        test_loss = 0
        results = []

        with torch.no_grad(), tqdm(
            total=len(dataloader),
            desc=desc or f"Valid Epoch {epoch_idx}/{self.max_epochs}",
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
                self.logger.log({"epoch": epoch_idx, **self.log_context, **metrics})
        return metrics

    def test(self, epoch_idx: int):
        return self.evaluate(
            self.test_dataloader,
            epoch_idx=epoch_idx,
            metric_prefix="valid",
            log=True,
        )

    def initialize_training(self, scheduler_epochs: int = None):
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self._optimizer_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_epochs or self.max_epochs,
            eta_min=0.0,
        )

    def fit(self):
        self.initialize_training()
        for epoch_idx in range(self.max_epochs):
            self.train_epoch(epoch_idx)
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

            self.scheduler.step()


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
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
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


def _stage_random_accuracy(config: ContinualTrainConfig, stage_idx: int):
    stage_config = config.data.test_stage_configs[stage_idx]
    random_accuracy = getattr(stage_config, "random_accuracy", None)
    if callable(random_accuracy):
        return float(random_accuracy())
    return None


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
        slice_keys=config.slice_keys,
        loss_type=config.loss_type,
        device=device,
        logger=logger,
    )
    trainer.initialize_training(
        scheduler_epochs=config.max_epochs * len(stage_dataloaders)
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

    should_stop = False
    for stage_idx, (train_dataloader, _) in enumerate(stage_dataloaders):
        trainer.train_dataloader = train_dataloader
        trainer.log_context = {"continual/stage": stage_idx}

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

        for local_epoch_idx in range(config.max_epochs):
            global_epoch_idx = stage_idx * config.max_epochs + local_epoch_idx
            trainer.train_epoch(global_epoch_idx)
            trainer.scheduler.step()

        stage_metrics = {}
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
        logger.log({
            "epoch": (stage_idx + 1) * config.max_epochs - 1,
            **metrics,
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
                "epoch": (stage_idx + 1) * config.max_epochs - 1,
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
