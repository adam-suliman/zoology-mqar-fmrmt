"""Selected formal CL reruns for compute-aware comparison.

This module reruns only the comparison-critical configs with timing metrics:

- 5-stage clean formal CL: Transformer, Base RMT n_mem=4, stable FMRMT.
- 10-stage horizon formal CL: Base RMT n_mem=4, Base RMT n_mem=16,
  stable FMRMT.

Seeds are fixed to ``{123, 456, 789}`` to match the existing formal artifacts.
"""

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.experiments.rmt_mqar._common import (
    base_rmt_model,
    fast_memory_rmt_model,
    transformer_model,
)


SEEDS = [123, 456, 789]
VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
BATCH_SIZE = 64
SEGMENT_LEN = 64
LEARNING_RATE = 3e-3

TASK_SPECS = {
    "5stage_clean": {
        "num_stages": 5,
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
    },
    "10stage_horizon": {
        "num_stages": 10,
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
    },
}


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


def stable_fmrmt_model():
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


def task_models(task_name: str):
    if task_name == "5stage_clean":
        return [
            transformer_model(
                input_seq_len=INPUT_SEQ_LEN,
                vocab_size=VOCAB_SIZE,
            ),
            base_rmt_model(
                input_seq_len=INPUT_SEQ_LEN,
                segment_len=SEGMENT_LEN,
                n_mem=4,
                vocab_size=VOCAB_SIZE,
            ).model_copy(update={"name": "base_rmt_nmem4"}),
            stable_fmrmt_model(),
        ]
    if task_name == "10stage_horizon":
        return [
            base_rmt_model(
                input_seq_len=INPUT_SEQ_LEN,
                segment_len=SEGMENT_LEN,
                n_mem=4,
                vocab_size=VOCAB_SIZE,
            ).model_copy(update={"name": "base_rmt_nmem4"}),
            base_rmt_model(
                input_seq_len=INPUT_SEQ_LEN,
                segment_len=SEGMENT_LEN,
                n_mem=16,
                vocab_size=VOCAB_SIZE,
            ).model_copy(update={"name": "base_rmt_nmem16"}),
            stable_fmrmt_model(),
        ]
    raise ValueError(f"Unknown task_name: {task_name}")


configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=model,
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "timing_selected",
                task_name,
            ]
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
            "input_seq_len",
        ],
        seed=seed,
        run_id=f"timing-{task_name}-{model.name}-seed{seed}",
    )
    for task_name in TASK_SPECS
    for seed in SEEDS
    for model in task_models(task_name)
]
