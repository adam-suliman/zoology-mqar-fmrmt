from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class ClassIncrementalARConfig(DataSegmentConfig):
    name: str = "class_incremental_ar"
    power_a: float = 0.01
    stage_idx: int = 0
    num_stages: int = 5
    associations_per_stage: int = 16
    num_query_associations: int = 8
    eval_mode: Literal["current", "seen"] = "current"
    distractor_mode: Literal["current", "seen", "all"] = "current"
    include_slices: bool = True

    def stage_range(self, stage_idx: int = None):
        stage_idx = self.stage_idx if stage_idx is None else stage_idx
        if not 0 <= stage_idx < self.num_stages:
            raise ValueError(f"stage_idx must be in [0, {self.num_stages})")
        return stage_idx, stage_idx + 1

    def active_stage_range(self):
        if self.eval_mode == "current":
            return self.stage_range()
        if self.eval_mode == "seen":
            return 0, self.stage_idx + 1
        raise ValueError(f"Unsupported eval_mode: {self.eval_mode}")

    def key_range(self, stage_idx: int = None):
        key_vocab_size = self.vocab_size // 2
        stage_start, stage_end = (
            self.stage_range(stage_idx)
            if stage_idx is not None
            else self.active_stage_range()
        )
        key_start = 1 + stage_start * self.associations_per_stage
        key_end = 1 + stage_end * self.associations_per_stage
        if key_end > key_vocab_size:
            raise ValueError(
                f"Key range [{key_start}, {key_end}) exceeds lower-half "
                f"vocab boundary {key_vocab_size}."
            )
        return key_start, key_end

    def value_range(self, stage_idx: int = None):
        value_vocab_start = self.vocab_size // 2
        stage_start, stage_end = (
            self.stage_range(stage_idx)
            if stage_idx is not None
            else self.active_stage_range()
        )
        value_start = value_vocab_start + stage_start * self.associations_per_stage
        value_end = value_vocab_start + stage_end * self.associations_per_stage
        if value_end > self.vocab_size:
            raise ValueError(
                f"Value range [{value_start}, {value_end}) exceeds vocab_size "
                f"{self.vocab_size}."
            )
        return value_start, value_end

    def random_accuracy(self):
        stage_start, stage_end = self.active_stage_range()
        active_values = (stage_end - stage_start) * self.associations_per_stage
        return 1.0 / active_values

    def build(self, seed: int) -> DataSegment:
        return class_incremental_ar(**self.model_dump(), seed=seed)


def _stage_range(stage_idx: int, num_stages: int):
    if not 0 <= stage_idx < num_stages:
        raise ValueError(f"stage_idx must be in [0, {num_stages})")
    return stage_idx, stage_idx + 1


def _active_stage_range(stage_idx: int, num_stages: int, eval_mode: str):
    if eval_mode == "current":
        return _stage_range(stage_idx, num_stages)
    if eval_mode == "seen":
        return 0, stage_idx + 1
    raise ValueError(f"Unsupported eval_mode: {eval_mode}")


def _association_tokens(
    vocab_size: int,
    associations_per_stage: int,
    stage_start: int,
    stage_end: int,
):
    association_offsets = np.arange(
        stage_start * associations_per_stage,
        stage_end * associations_per_stage,
    )
    key_vocab_size = vocab_size // 2
    keys = 1 + association_offsets
    values = key_vocab_size + association_offsets
    return keys, values


def _distractor_stage_range(stage_idx: int, num_stages: int, distractor_mode: str):
    if distractor_mode == "current":
        return _stage_range(stage_idx, num_stages)
    if distractor_mode == "seen":
        return 0, stage_idx + 1
    if distractor_mode == "all":
        return 0, num_stages
    raise ValueError(f"Unsupported distractor_mode: {distractor_mode}")


def class_incremental_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    stage_idx: int = 0,
    num_stages: int = 5,
    associations_per_stage: int = 16,
    num_query_associations: int = 8,
    eval_mode: str = "current",
    distractor_mode: str = "current",
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """
    Class-incremental associative retrieval.

    Each stage introduces a fixed disjoint block of key-value associations. For
    stage-local training, examples sample from only the current block. For
    cumulative evaluation, examples sample from all blocks seen up to stage_idx.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert 0 <= stage_idx < num_stages

    key_vocab_size = vocab_size // 2
    total_associations = num_stages * associations_per_stage
    if 1 + total_associations > key_vocab_size:
        raise ValueError(
            f"{total_associations} key associations do not fit below "
            f"lower-half vocab boundary {key_vocab_size} with token 0 reserved."
        )
    if key_vocab_size + total_associations > vocab_size:
        raise ValueError(
            f"{total_associations} value associations do not fit in upper-half "
            f"vocab ending at {vocab_size}."
        )

    active_stage_start, active_stage_end = _active_stage_range(
        stage_idx=stage_idx,
        num_stages=num_stages,
        eval_mode=eval_mode,
    )
    active_associations = (
        active_stage_end - active_stage_start
    ) * associations_per_stage
    assert num_query_associations <= active_associations

    context_size = num_query_associations * 2
    assert context_size + num_query_associations * 2 <= input_seq_len

    np.random.seed(seed)

    key_choices, value_choices = _association_tokens(
        vocab_size=vocab_size,
        associations_per_stage=associations_per_stage,
        stage_start=active_stage_start,
        stage_end=active_stage_end,
    )

    association_ids = np.arange(active_associations)
    selected = np.stack(
        [
            np.random.choice(
                association_ids,
                replace=False,
                size=num_query_associations,
            )
            for _ in range(num_examples)
        ]
    )
    keys = key_choices[selected]
    values = value_choices[selected]

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice,
        axis=1,
        arr=x,
        replace=False,
        p=p,
        size=num_query_associations,
    )

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(
        labels,
        (gaps * 2) + context_size + 1,
        values=values,
        axis=1,
    )

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    distractor_stage_start, distractor_stage_end = _distractor_stage_range(
        stage_idx=stage_idx,
        num_stages=num_stages,
        distractor_mode=distractor_mode,
    )
    distractor_keys, distractor_values = _association_tokens(
        vocab_size=vocab_size,
        associations_per_stage=associations_per_stage,
        stage_start=distractor_stage_start,
        stage_end=distractor_stage_end,
    )
    distractor_choices = torch.tensor(
        np.concatenate([distractor_keys, distractor_values]),
        dtype=inputs.dtype,
    )
    zero_mask = inputs == 0
    random_indices = torch.randint(
        len(distractor_choices),
        size=inputs.shape,
        device=inputs.device,
    )
    inputs[zero_mask] = distractor_choices[random_indices][zero_mask]

    slices = {}
    if include_slices:
        current_key_start = 1 + stage_idx * associations_per_stage
        current_key_end = current_key_start + associations_per_stage
        current_value_start = key_vocab_size + stage_idx * associations_per_stage
        current_value_end = current_value_start + associations_per_stage
        active_key_start = 1 + active_stage_start * associations_per_stage
        active_key_end = 1 + active_stage_end * associations_per_stage
        active_value_start = key_vocab_size + active_stage_start * associations_per_stage
        active_value_end = key_vocab_size + active_stage_end * associations_per_stage
        slices = {
            "stage_idx": stage_idx,
            "num_stages": num_stages,
            "associations_per_stage": associations_per_stage,
            "num_query_associations": num_query_associations,
            "input_seq_len": input_seq_len,
            "eval_mode": eval_mode,
            "distractor_mode": distractor_mode,
            "active_stage_start": active_stage_start,
            "active_stage_end": active_stage_end,
            "current_key_start": current_key_start,
            "current_key_end": current_key_end,
            "current_value_start": current_value_start,
            "current_value_end": current_value_end,
            "active_key_start": active_key_start,
            "active_key_end": active_key_end,
            "active_value_start": active_value_start,
            "active_value_end": active_value_end,
            "random_accuracy": 1.0 / active_associations,
        }

    return DataSegment(inputs, labels, slices=slices)
