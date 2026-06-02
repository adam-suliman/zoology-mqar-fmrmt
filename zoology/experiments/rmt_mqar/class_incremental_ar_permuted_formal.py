"""Formal fixed-random-permutation class-incremental AR runs.

This module reuses the short permuted AR runner's JSON capture logic, but adds
paper-scale 5/10/20-stage task specs and stronger model choices.

Default run:

```
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_permuted_formal
```

Useful overrides:

```
PERMUTED_AR_FORMAL_TASKS=formal10_permuted,formal20_permuted
PERMUTED_AR_FORMAL_SEEDS=123,456,789
PERMUTED_AR_FORMAL_MODELS=attention,base_rmt_nmem16,fmrmt_lr0p005_slow2,fmrmt_fast0_slow2
PERMUTED_AR_FORMAL_LR_SCHEDULER_MODE=stage_onecycle
PERMUTED_AR_FORMAL_SLOW_UPDATE_MODE=accumulate
PERMUTED_AR_FORMAL_SEGMENT_LEN=128
PERMUTED_AR_FORMAL_INPUT_SEQ_LEN=256
PERMUTED_AR_FORMAL_VOCAB_SIZE=2048
PERMUTED_AR_FORMAL_ASSOCIATIONS_PER_STAGE=32
PERMUTED_AR_FORMAL_NUM_QUERY_ASSOCIATIONS=16
```
"""

from __future__ import annotations

import os
import random
import re

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.experiments.rmt_mqar import class_incremental_ar_permuted_smoke as base
from zoology.experiments.rmt_mqar._common import base_rmt_model, fast_memory_rmt_model, transformer_model


FORMAL_TASK_SPECS = {
    "formal5_permuted": {
        "num_stages": 5,
        "train_examples": 4096,
        "test_examples": 1024,
        "max_epochs": 16,
    },
    "formal10_permuted": {
        "num_stages": 10,
        "train_examples": 2048,
        "test_examples": 512,
        "max_epochs": 8,
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


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


VOCAB_SIZE = _int_env("PERMUTED_AR_FORMAL_VOCAB_SIZE", base.VOCAB_SIZE)
ASSOCIATIONS_PER_STAGE = _int_env(
    "PERMUTED_AR_FORMAL_ASSOCIATIONS_PER_STAGE",
    base.ASSOCIATIONS_PER_STAGE,
)
NUM_QUERY_ASSOCIATIONS = _int_env(
    "PERMUTED_AR_FORMAL_NUM_QUERY_ASSOCIATIONS",
    base.NUM_QUERY_ASSOCIATIONS,
)
INPUT_SEQ_LEN = _int_env("PERMUTED_AR_FORMAL_INPUT_SEQ_LEN", base.INPUT_SEQ_LEN)
BATCH_SIZE = _int_env("PERMUTED_AR_FORMAL_BATCH_SIZE", base.BATCH_SIZE)
LEARNING_RATE = _float_env("PERMUTED_AR_FORMAL_LEARNING_RATE", base.LEARNING_RATE)
TRAIN_EXAMPLES_OVERRIDE = _optional_int_env("PERMUTED_AR_FORMAL_TRAIN_EXAMPLES")
TEST_EXAMPLES_OVERRIDE = _optional_int_env("PERMUTED_AR_FORMAL_TEST_EXAMPLES")
MAX_EPOCHS_OVERRIDE = _optional_int_env("PERMUTED_AR_FORMAL_MAX_EPOCHS")

base.VOCAB_SIZE = VOCAB_SIZE
base.ASSOCIATIONS_PER_STAGE = ASSOCIATIONS_PER_STAGE
base.NUM_QUERY_ASSOCIATIONS = NUM_QUERY_ASSOCIATIONS
base.INPUT_SEQ_LEN = INPUT_SEQ_LEN
base.BATCH_SIZE = BATCH_SIZE
base.LEARNING_RATE = LEARNING_RATE

for task_spec in FORMAL_TASK_SPECS.values():
    if TRAIN_EXAMPLES_OVERRIDE is not None:
        task_spec["train_examples"] = TRAIN_EXAMPLES_OVERRIDE
    if TEST_EXAMPLES_OVERRIDE is not None:
        task_spec["test_examples"] = TEST_EXAMPLES_OVERRIDE
    if MAX_EPOCHS_OVERRIDE is not None:
        task_spec["max_epochs"] = MAX_EPOCHS_OVERRIDE


SEEDS = _csv_env("PERMUTED_AR_FORMAL_SEEDS", [123, 456, 789], int)
TASK_NAMES = _csv_env("PERMUTED_AR_FORMAL_TASKS", ["formal20_permuted"], str)
MODEL_KEYS = _csv_env(
    "PERMUTED_AR_FORMAL_MODELS",
    ["attention", "base_rmt_nmem16", "fmrmt_lr0p005_slow2", "fmrmt_fast0_slow2"],
    str,
)
EPOCH_EVAL_INTERVAL = int(os.getenv("PERMUTED_AR_FORMAL_EPOCH_EVAL_INTERVAL", "1"))
LR_SCHEDULER_MODE = os.getenv("PERMUTED_AR_FORMAL_LR_SCHEDULER_MODE", "stage_onecycle")
SLOW_UPDATE_MODE = os.getenv("PERMUTED_AR_FORMAL_SLOW_UPDATE_MODE", "accumulate")
STAGE_ORDER_SPEC = os.getenv("PERMUTED_AR_FORMAL_STAGE_ORDER", "sequential")
OUTPUT_PREFIX_SUFFIX = os.getenv("PERMUTED_AR_FORMAL_OUTPUT_PREFIX_SUFFIX", "")
SEGMENT_LEN = int(os.getenv("PERMUTED_AR_FORMAL_SEGMENT_LEN", str(base.SEGMENT_LEN)))
if SLOW_UPDATE_MODE not in {"skip", "accumulate"}:
    raise ValueError(
        "PERMUTED_AR_FORMAL_SLOW_UPDATE_MODE must be 'skip' or 'accumulate'"
    )

base.TASK_SPECS.update(FORMAL_TASK_SPECS)
unknown_tasks = sorted(set(TASK_NAMES) - set(base.TASK_SPECS))
if unknown_tasks:
    raise ValueError(f"Unknown PERMUTED_AR_FORMAL_TASKS: {unknown_tasks}")


FMRMT_SLOW2_FAST_LR_KEYS = {
    "fmrmt_fast0_slow2": 0.0,
    "fmrmt_lr0p001_slow2": 0.001,
    "fmrmt_lr0p002_slow2": 0.002,
    "fmrmt_lr0p005_slow2": 0.005,
    "fmrmt_lr0p01_slow2": 0.01,
    "fmrmt_lr0p02_slow2": 0.02,
    "fmrmt_lr0p05_slow2": 0.05,
    "fmrmt_lr0p1_slow2": 0.1,
    "fmrmt_lr0p2_slow2": 0.2,
}

BASE_RMT_MODEL_KEY = re.compile(r"^base_rmt_nmem(?P<n_mem>\d+)(?P<bptt>_bptt)?$")
FMRMT_MODEL_KEY = re.compile(
    r"^fmrmt(?:_nmem(?P<n_mem>\d+))?_"
    r"(?P<fast_lr>fast0|lr\d+(?:p\d+)?)_"
    r"slow(?P<slow_update_freq>\d+)"
    r"(?P<suffixes>(?:_[a-z0-9]+)*)$"
)


def _fast_lr_from_token(token: str) -> float:
    if token == "fast0":
        return 0.0
    if token.startswith("lr"):
        return float(token[2:].replace("p", "."))
    raise ValueError(f"Unsupported FastMem LR token: {token}")


def _suffix_tokens(suffixes: str):
    if not suffixes:
        return []
    return [token for token in suffixes.split("_") if token]


def stage_order_for_task(task_name: str):
    num_stages = base.TASK_SPECS[task_name]["num_stages"]
    raw = STAGE_ORDER_SPEC.strip()
    if raw in {"", "sequential", "identity"}:
        return list(range(num_stages))
    if raw == "reverse":
        return list(reversed(range(num_stages)))
    if raw.startswith("shuffle:"):
        seed = int(raw.split(":", 1)[1])
        order = list(range(num_stages))
        random.Random(seed).shuffle(order)
    else:
        order = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if sorted(order) != list(range(num_stages)):
        raise ValueError(
            f"PERMUTED_AR_FORMAL_STAGE_ORDER must be a permutation of "
            f"0..{num_stages - 1}; got {order}"
        )
    return order


def data_config(task_name: str, seed: int):
    spec = base.TASK_SPECS[task_name]
    num_stages = spec["num_stages"]
    stage_order = stage_order_for_task(task_name)
    return ContinualDataConfig(
        train_stage_configs=[
            base.stage_config(
                stage_idx=actual_stage_idx,
                num_stages=num_stages,
                num_examples=spec["train_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for actual_stage_idx in stage_order
        ],
        test_stage_configs=[
            base.stage_config(
                stage_idx=actual_stage_idx,
                num_stages=num_stages,
                num_examples=spec["test_examples"],
                eval_mode="current",
                distractor_mode="current",
            )
            for actual_stage_idx in stage_order
        ],
        batch_size=base.BATCH_SIZE,
        seed=seed,
    )


def fmrmt_slow2_model(
    model_key: str,
    eval_memory_policy: str = "initial",
    reset_memory_each_epoch: bool = True,
    detach_memory_between_segments: bool = False,
    n_mem: int = 8,
):
    return fast_memory_rmt_model(
        input_seq_len=base.INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        n_mem=n_mem,
        rmt_fast_lr=FMRMT_SLOW2_FAST_LR_KEYS[model_key],
        rmt_slow_update_freq=2,
        detach_memory_between_segments=detach_memory_between_segments,
        reset_memory_each_batch=False,
        reset_memory_each_epoch=reset_memory_each_epoch,
        rmt_clip_memory_grad=1.0,
        eval_memory_policy=eval_memory_policy,
        vocab_size=base.VOCAB_SIZE,
    ).model_copy(update={"name": model_key})


def dynamic_base_rmt_model(model_key: str):
    match = BASE_RMT_MODEL_KEY.fullmatch(model_key)
    if match is None:
        return None
    n_mem = int(match.group("n_mem"))
    detach_memory_between_segments = match.group("bptt") is None
    return base_rmt_model(
        input_seq_len=base.INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        n_mem=n_mem,
        detach_memory_between_segments=detach_memory_between_segments,
        vocab_size=base.VOCAB_SIZE,
    ).model_copy(update={"name": model_key})


def dynamic_fmrmt_model(model_key: str):
    match = FMRMT_MODEL_KEY.fullmatch(model_key)
    if match is None:
        return None

    n_mem = int(match.group("n_mem") or 8)
    rmt_fast_lr = _fast_lr_from_token(match.group("fast_lr"))
    rmt_slow_update_freq = int(match.group("slow_update_freq"))
    detach_memory_between_segments = False
    eval_memory_policy = "initial"
    reset_memory_each_batch = False
    reset_memory_each_epoch = True

    suffixes = set(_suffix_tokens(match.group("suffixes")))
    unknown_suffixes = set(suffixes)
    if "detach" in suffixes or "detachtrue" in suffixes:
        detach_memory_between_segments = True
        unknown_suffixes.discard("detach")
        unknown_suffixes.discard("detachtrue")
    if "bptt" in suffixes or "detachfalse" in suffixes:
        detach_memory_between_segments = False
        unknown_suffixes.discard("bptt")
        unknown_suffixes.discard("detachfalse")
    if "evalfast" in suffixes:
        eval_memory_policy = "fast"
        unknown_suffixes.discard("evalfast")
    if "evalinitial" in suffixes:
        eval_memory_policy = "initial"
        unknown_suffixes.discard("evalinitial")
    if "noepochreset" in suffixes:
        reset_memory_each_epoch = False
        unknown_suffixes.discard("noepochreset")
    if "epochreset" in suffixes:
        reset_memory_each_epoch = True
        unknown_suffixes.discard("epochreset")
    if "batchreset" in suffixes:
        reset_memory_each_batch = True
        unknown_suffixes.discard("batchreset")
    if "nobatchreset" in suffixes:
        reset_memory_each_batch = False
        unknown_suffixes.discard("nobatchreset")
    if unknown_suffixes:
        raise ValueError(
            f"Unknown FastMem-RMT suffixes for {model_key}: "
            f"{sorted(unknown_suffixes)}"
        )

    return fast_memory_rmt_model(
        input_seq_len=base.INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        n_mem=n_mem,
        rmt_fast_lr=rmt_fast_lr,
        rmt_slow_update_freq=rmt_slow_update_freq,
        detach_memory_between_segments=detach_memory_between_segments,
        reset_memory_each_batch=reset_memory_each_batch,
        reset_memory_each_epoch=reset_memory_each_epoch,
        rmt_clip_memory_grad=1.0,
        eval_memory_policy=eval_memory_policy,
        vocab_size=base.VOCAB_SIZE,
    ).model_copy(update={"name": model_key})


def model_by_key(model_key: str):
    dynamic_model = dynamic_base_rmt_model(model_key)
    if dynamic_model is not None:
        return dynamic_model
    dynamic_model = dynamic_fmrmt_model(model_key)
    if dynamic_model is not None:
        return dynamic_model
    if model_key == "attention":
        return transformer_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            vocab_size=base.VOCAB_SIZE,
        )
    if model_key == "base_rmt_nmem4":
        return base_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=4,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem4"})
    if model_key == "base_rmt_nmem8":
        return base_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem8"})
    if model_key == "base_rmt_nmem16":
        return base_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=16,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem16"})
    if model_key == "base_rmt_nmem16_bptt":
        return base_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=16,
            detach_memory_between_segments=False,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "base_rmt_nmem16_bptt"})
    if model_key == "fmrmt_stable":
        return fast_memory_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.005,
            rmt_slow_update_freq=4,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_stable_lr0p005_slow4"})
    if model_key == "fmrmt_stable_evalfast":
        return fast_memory_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.005,
            rmt_slow_update_freq=4,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            eval_memory_policy="fast",
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_stable_evalfast"})
    if model_key == "fmrmt_fast0_slow4":
        return fast_memory_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.0,
            rmt_slow_update_freq=4,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_fast0_slow4"})
    if model_key in FMRMT_SLOW2_FAST_LR_KEYS:
        return fmrmt_slow2_model(model_key)
    if model_key == "fmrmt_lr0p005_slow2_evalfast":
        return fmrmt_slow2_model("fmrmt_lr0p005_slow2", eval_memory_policy="fast").model_copy(
            update={"name": "fmrmt_lr0p005_slow2_evalfast"}
        )
    if model_key == "fmrmt_lr0p005_slow2_noepochreset":
        return fmrmt_slow2_model(
            "fmrmt_lr0p005_slow2",
            reset_memory_each_epoch=False,
        ).model_copy(update={"name": "fmrmt_lr0p005_slow2_noepochreset"})
    if model_key == "fmrmt_lr0p005_slow2_noepochreset_evalfast":
        return fmrmt_slow2_model(
            "fmrmt_lr0p005_slow2",
            eval_memory_policy="fast",
            reset_memory_each_epoch=False,
        ).model_copy(update={"name": "fmrmt_lr0p005_slow2_noepochreset_evalfast"})
    if model_key == "fmrmt_plastic":
        return fast_memory_rmt_model(
            input_seq_len=base.INPUT_SEQ_LEN,
            segment_len=SEGMENT_LEN,
            n_mem=8,
            rmt_fast_lr=0.01,
            rmt_slow_update_freq=1,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=base.VOCAB_SIZE,
        ).model_copy(update={"name": "fmrmt_plastic_lr0p01_slow1"})
    raise ValueError(f"Unknown PERMUTED_AR_FORMAL_MODELS entry: {model_key}")


base.SEEDS = SEEDS
base.TASK_NAMES = TASK_NAMES
base.MODEL_KEYS = MODEL_KEYS
base.SEGMENT_LEN = SEGMENT_LEN
base.model_by_key = model_by_key
base.configs = [
    ContinualTrainConfig(
        data=data_config(task_name=task_name, seed=seed),
        model=model_by_key(model_key),
        logger=LoggerConfig(
            tags=[
                "rmt_mqar",
                "class_incremental_ar",
                "permuted_mapping",
                "formal",
                task_name,
            ]
        ),
        max_epochs=base.TASK_SPECS[task_name]["max_epochs"],
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
        run_id=f"permuted-smoke-{task_name}-{model_key}-seed{seed}",
    )
    for task_name in TASK_NAMES
    for seed in SEEDS
    for model_key in MODEL_KEYS
]


def run_all():
    return base.run_all(
        output_prefix=(
            "class_incremental_ar_permuted_formal_"
            f"{LR_SCHEDULER_MODE}_{SLOW_UPDATE_MODE}"
            f"{OUTPUT_PREFIX_SUFFIX}"
        )
    )


if __name__ == "__main__":
    run_all()
