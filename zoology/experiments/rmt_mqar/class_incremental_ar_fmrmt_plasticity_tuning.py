"""FMRMT plasticity tuning for hard formal CL associative-retrieval cases.

Default grid is a one-seed screen over both hard cases:

- 10-stage class-incremental horizon.
- 5-stage repeated-key latest-value interference.

It sweeps:

- ``rmt_fast_lr in {0.001, 0.005, 0.01, 0.02}``
- ``rmt_slow_update_freq in {1, 2, 4}``
- ``reset_memory_each_epoch in {True, False}``

``reset_memory_each_batch`` stays ``False``. Use environment variables to narrow
or replicate the grid before launching, for example:

``FMRMT_TUNING_SEEDS=123,456,789``
``FMRMT_TUNING_TASKS=horizon10``
``FMRMT_TUNING_FAST_LRS=0.005,0.01``
``FMRMT_TUNING_SLOW_FREQS=2,4``
``FMRMT_TUNING_EPOCH_RESETS=true,false``
"""

import os

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.data.interference_ar import InterferenceARConfig
from zoology.experiments.rmt_mqar._common import fast_memory_rmt_model


VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
TRAIN_EXAMPLES = 2_048
TEST_EXAMPLES = 512
MAX_EPOCHS = 8
BATCH_SIZE = 64
SEGMENT_LEN = 64
N_MEM = 8
RMT_CLIP_MEMORY_GRAD = 1.0
RESET_MEMORY_EACH_BATCH = False
LEARNING_RATE = 3e-3

PRIMARY_METRICS = [
    "continual/plasticity",
    "continual/avg_learning_accuracy",
    "continual/avg_bwt",
    "continual/avg_forgetting_from_learning",
    "continual/seen_avg_accuracy",
]


def _csv_env(name: str, default, cast):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _float_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


SEEDS = _csv_env("FMRMT_TUNING_SEEDS", [123], int)
FAST_LRS = _csv_env("FMRMT_TUNING_FAST_LRS", [0.001, 0.005, 0.01, 0.02], float)
SLOW_UPDATE_FREQS = _csv_env("FMRMT_TUNING_SLOW_FREQS", [1, 2, 4], int)
RESET_MEMORY_EACH_EPOCH_OPTIONS = _csv_env(
    "FMRMT_TUNING_EPOCH_RESETS",
    [True, False],
    _bool,
)


def horizon_stage_config(
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


def horizon10_data_config(seed: int):
    num_stages = 10
    return ContinualDataConfig(
        train_stage_configs=[
            horizon_stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=TRAIN_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        test_stage_configs=[
            horizon_stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=TEST_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


def interference_stage_config(
    stage_idx: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
):
    return InterferenceARConfig(
        stage_idx=stage_idx,
        num_stages=5,
        associations_per_stage=ASSOCIATIONS_PER_STAGE,
        num_query_associations=NUM_QUERY_ASSOCIATIONS,
        num_interference_pairs=4,
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        target_policy="latest",
        include_slices=True,
    )


def latest_updates4_data_config(seed: int):
    num_stages = 5
    return ContinualDataConfig(
        train_stage_configs=[
            interference_stage_config(
                stage_idx=stage_idx,
                num_examples=TRAIN_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        test_stage_configs=[
            interference_stage_config(
                stage_idx=stage_idx,
                num_examples=TEST_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


TASK_BUILDERS = {
    "horizon10": horizon10_data_config,
    "latest_updates4": latest_updates4_data_config,
}
TASK_NAMES = _csv_env("FMRMT_TUNING_TASKS", TASK_BUILDERS.keys(), str)
UNKNOWN_TASKS = sorted(set(TASK_NAMES) - set(TASK_BUILDERS))
if UNKNOWN_TASKS:
    raise ValueError(f"Unknown FMRMT tuning tasks: {UNKNOWN_TASKS}")


def tuning_model(
    rmt_fast_lr: float,
    rmt_slow_update_freq: int,
    reset_memory_each_epoch: bool,
):
    reset_name = "epochreset" if reset_memory_each_epoch else "noepochreset"
    name = (
        f"fmrmt_lr{_float_slug(rmt_fast_lr)}_"
        f"slow{rmt_slow_update_freq}_{reset_name}"
    )
    return fast_memory_rmt_model(
        input_seq_len=INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        n_mem=N_MEM,
        rmt_fast_lr=rmt_fast_lr,
        rmt_slow_update_freq=rmt_slow_update_freq,
        rmt_clip_memory_grad=RMT_CLIP_MEMORY_GRAD,
        reset_memory_each_batch=RESET_MEMORY_EACH_BATCH,
        reset_memory_each_epoch=reset_memory_each_epoch,
        vocab_size=VOCAB_SIZE,
    ).model_copy(update={"name": name})


def slice_keys_for_task(task_name: str):
    keys = [
        "stage_idx",
        "eval_mode",
        "associations_per_stage",
        "num_query_associations",
        "input_seq_len",
    ]
    if task_name == "latest_updates4":
        keys.extend(["num_interference_pairs", "target_policy"])
    return keys


configs = [
    ContinualTrainConfig(
        data=TASK_BUILDERS[task_name](seed=seed),
        model=tuning_model(
            rmt_fast_lr=rmt_fast_lr,
            rmt_slow_update_freq=rmt_slow_update_freq,
            reset_memory_each_epoch=reset_memory_each_epoch,
        ),
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "fmrmt_plasticity_tuning",
                task_name,
            ]
        ),
        max_epochs=MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        slice_keys=slice_keys_for_task(task_name),
        seed=seed,
        run_id=(
            f"fmrmt-plasticity-{task_name}-"
            f"lr{_float_slug(rmt_fast_lr)}-"
            f"slow{rmt_slow_update_freq}-"
            f"epochreset{int(reset_memory_each_epoch)}-"
            f"seed{seed}"
        ),
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for rmt_fast_lr in FAST_LRS
    for rmt_slow_update_freq in SLOW_UPDATE_FREQS
    for reset_memory_each_epoch in RESET_MEMORY_EACH_EPOCH_OPTIONS
]
