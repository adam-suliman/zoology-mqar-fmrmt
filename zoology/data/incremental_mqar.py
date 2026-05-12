import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class IncrementalMQARConfig(DataSegmentConfig):
    name: str = "incremental_mqar"
    power_a: float = 0.01
    stage_idx: int = 0
    num_stages: int = 4
    keys_per_stage: int = 128
    values_per_stage: int = 128
    num_kv_pairs: int = 8
    random_non_queries: bool = True
    include_slices: bool = True

    def key_range(self):
        key_vocab_size = self.vocab_size // 2
        key_start = 1 + self.stage_idx * self.keys_per_stage
        key_end = key_start + self.keys_per_stage
        if key_end > key_vocab_size:
            raise ValueError(
                f"Stage {self.stage_idx} key range [{key_start}, {key_end}) "
                f"exceeds lower-half vocab boundary {key_vocab_size}."
            )
        return key_start, key_end

    def value_range(self):
        value_vocab_start = self.vocab_size // 2
        value_start = value_vocab_start + self.stage_idx * self.values_per_stage
        value_end = value_start + self.values_per_stage
        if value_end > self.vocab_size:
            raise ValueError(
                f"Stage {self.stage_idx} value range [{value_start}, {value_end}) "
                f"exceeds vocab_size {self.vocab_size}."
            )
        return value_start, value_end

    def random_accuracy(self):
        return 1.0 / self.values_per_stage

    def build(self, seed: int) -> DataSegment:
        return incremental_mqar(**self.model_dump(), seed=seed)


def incremental_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    stage_idx: int = 0,
    num_stages: int = 4,
    keys_per_stage: int = 128,
    values_per_stage: int = 128,
    num_kv_pairs: int = 8,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """
    Stage-local MQAR for continual vocab-growth experiments.

    Each stage samples keys from one disjoint lower-half vocabulary range and
    values from one disjoint upper-half vocabulary range.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert 0 <= stage_idx < num_stages
    assert num_kv_pairs <= keys_per_stage
    assert num_kv_pairs <= values_per_stage

    context_size = num_kv_pairs * 2
    assert context_size + num_kv_pairs * 2 <= input_seq_len

    key_vocab_size = vocab_size // 2
    key_start = 1 + stage_idx * keys_per_stage
    key_end = key_start + keys_per_stage
    value_start = key_vocab_size + stage_idx * values_per_stage
    value_end = value_start + values_per_stage
    assert key_end <= key_vocab_size
    assert value_end <= vocab_size

    np.random.seed(seed)

    key_choices = np.arange(key_start, key_end)
    value_choices = np.arange(value_start, value_end)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice,
        1,
        keys_unshuffled,
        replace=False,
        size=num_kv_pairs,
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice,
        1,
        values_unshuffled,
        replace=False,
        size=num_kv_pairs,
    )

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
        size=num_kv_pairs,
    )

    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    if random_non_queries:
        distractor_choices = torch.tensor(
            np.concatenate([key_choices, value_choices]),
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
        slices = {
            "stage_idx": stage_idx,
            "num_stages": num_stages,
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "key_start": key_start,
            "key_end": key_end,
            "value_start": value_start,
            "value_end": value_end,
        }

    return DataSegment(inputs, labels, slices=slices)
