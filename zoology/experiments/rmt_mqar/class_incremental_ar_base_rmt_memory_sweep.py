"""Base RMT memory-token sweep for formal class-incremental AR controls.

Default grid uses the same seeds as the existing formal CL artifacts so runs can
be compared directly:

- ``n_mem in {2, 4, 8, 16}``
- seeds ``{123, 456, 789}``
- clean 5-stage formal CL budget from ``class_incremental_ar.py``
- clean 10-stage horizon budget from ``class_incremental_ar_horizon.py``

Optional environment filters:

``BASE_RMT_MEMORY_SWEEP_TASKS=5stage,10stage``
``BASE_RMT_MEMORY_SWEEP_N_MEMS=2,4,8,16``
``BASE_RMT_MEMORY_SWEEP_SEEDS=123,456,789``
"""

import os

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.experiments.rmt_mqar._common import base_rmt_model


VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
BATCH_SIZE = 64
SEGMENT_LEN = 64
LEARNING_RATE = 3e-3

TASK_SPECS = {
    "5stage": {
        "num_stages": 5,
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
    },
    "10stage": {
        "num_stages": 10,
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


TASK_NAMES = _csv_env("BASE_RMT_MEMORY_SWEEP_TASKS", TASK_SPECS.keys(), str)
N_MEMS = _csv_env("BASE_RMT_MEMORY_SWEEP_N_MEMS", [2, 4, 8, 16], int)
SEEDS = _csv_env("BASE_RMT_MEMORY_SWEEP_SEEDS", [123, 456, 789], int)

UNKNOWN_TASKS = sorted(set(TASK_NAMES) - set(TASK_SPECS))
if UNKNOWN_TASKS:
    raise ValueError(f"Unknown Base RMT memory-sweep tasks: {UNKNOWN_TASKS}")


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


def memory_model(n_mem: int):
    return base_rmt_model(
        input_seq_len=INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        n_mem=n_mem,
        vocab_size=VOCAB_SIZE,
    ).model_copy(update={"name": f"base_rmt_nmem{n_mem}"})


configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=memory_model(n_mem=n_mem),
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "base_rmt_memory_sweep",
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
        run_id=f"base-rmt-memory-{task_name}-nmem{n_mem}-seed{seed}",
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for n_mem in N_MEMS
]
