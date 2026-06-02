import importlib.util

import torch

from zoology.config import ContinualTrainConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig


def tiny_config(
    stage_idx: int,
    eval_mode: str = "current",
    distractor_mode: str = "current",
    num_examples: int = 8,
    num_query_associations: int = 3,
    value_mapping: str = "aligned",
    association_table_seed: int = 0,
):
    return ClassIncrementalARConfig(
        stage_idx=stage_idx,
        num_stages=4,
        associations_per_stage=4,
        num_query_associations=num_query_associations,
        input_seq_len=64,
        vocab_size=128,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        value_mapping=value_mapping,
        association_table_seed=association_table_seed,
        include_slices=True,
    )


def non_ignored_labels(data):
    return data.labels[data.labels != -100]


def context_mapping(data, config):
    context_size = config.num_query_associations * 2
    context = data.inputs[:, :context_size]
    mapping = {}
    for row in context:
        for key, value in zip(row[0::2].tolist(), row[1::2].tolist()):
            if key in mapping:
                assert mapping[key] == value
            mapping[key] = value
    return mapping


def test_class_incremental_ar_stage_ranges_are_disjoint():
    stage0 = tiny_config(stage_idx=0)
    stage1 = tiny_config(stage_idx=1)

    assert stage0.key_range(stage_idx=0)[1] <= stage1.key_range(stage_idx=1)[0]
    assert stage0.value_range(stage_idx=0)[1] <= stage1.value_range(stage_idx=1)[0]


def test_class_incremental_ar_current_mode_labels_are_current_stage_only():
    config = tiny_config(stage_idx=2, eval_mode="current", num_examples=16)
    data = config.build(seed=123)
    value_start, value_end = config.value_range(stage_idx=2)
    labels = non_ignored_labels(data)

    assert data.inputs.shape == (16, 64)
    assert data.labels.shape == (16, 64)
    assert labels.numel() > 0
    assert torch.all((value_start <= labels) & (labels < value_end))
    assert data.slices["eval_mode"] == "current"
    assert data.slices["active_stage_start"] == 2
    assert data.slices["active_stage_end"] == 3


def test_class_incremental_ar_seen_mode_labels_cover_seen_stages():
    config = tiny_config(
        stage_idx=2,
        eval_mode="seen",
        distractor_mode="seen",
        num_examples=4,
        num_query_associations=12,
    )
    data = config.build(seed=123)
    active_value_start, active_value_end = config.value_range()
    current_value_start, current_value_end = config.value_range(stage_idx=2)
    labels = non_ignored_labels(data)

    assert torch.all((active_value_start <= labels) & (labels < active_value_end))
    assert torch.any(labels < current_value_start)
    assert torch.any((current_value_start <= labels) & (labels < current_value_end))
    assert data.slices["eval_mode"] == "seen"
    assert data.slices["active_stage_start"] == 0
    assert data.slices["active_stage_end"] == 3


def test_class_incremental_ar_permuted_mapping_is_stable_across_data_seeds():
    config = tiny_config(
        stage_idx=0,
        num_examples=16,
        num_query_associations=4,
        value_mapping="permuted",
        association_table_seed=0,
    )
    first = config.build(seed=123)
    second = config.build(seed=456)
    first_mapping = context_mapping(first, config)
    second_mapping = context_mapping(second, config)
    value_start, value_end = config.value_range(stage_idx=0)

    assert first_mapping == second_mapping
    assert sorted(first_mapping.values()) == list(range(value_start, value_end))
    assert any(
        value - value_start != key - config.key_range(stage_idx=0)[0]
        for key, value in first_mapping.items()
    )
    assert first.slices["value_mapping"] == "permuted"
    assert first.slices["association_table_seed"] == 0


def test_class_incremental_ar_seed_controls_distractor_fill():
    config = tiny_config(stage_idx=1, num_examples=16)
    torch.manual_seed(1)
    first = config.build(seed=123)
    torch.manual_seed(2)
    second = config.build(seed=123)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.labels, second.labels)


def test_class_incremental_ar_final_seen_random_baseline():
    config = tiny_config(
        stage_idx=3,
        eval_mode="seen",
        distractor_mode="seen",
        num_query_associations=16,
    )
    data = config.build(seed=123)

    assert config.random_accuracy() == 1.0 / (
        config.num_stages * config.associations_per_stage
    )
    assert data.slices["random_accuracy"] == config.random_accuracy()


def test_class_incremental_ar_experiment_imports_configs():
    spec = importlib.util.spec_from_file_location(
        "class_incremental_ar",
        "zoology/experiments/rmt_mqar/class_incremental_ar.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module.configs) == 9
    assert all(isinstance(config, ContinualTrainConfig) for config in module.configs)
    assert all(config.evaluate_future_stages for config in module.configs)


def test_class_incremental_ar_horizon_experiment_imports_configs():
    spec = importlib.util.spec_from_file_location(
        "class_incremental_ar_horizon",
        "zoology/experiments/rmt_mqar/class_incremental_ar_horizon.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module.configs) == 27
    assert all(isinstance(config, ContinualTrainConfig) for config in module.configs)
    assert all(config.evaluate_future_stages for config in module.configs)


if __name__ == "__main__":
    test_class_incremental_ar_stage_ranges_are_disjoint()
    test_class_incremental_ar_current_mode_labels_are_current_stage_only()
    test_class_incremental_ar_seen_mode_labels_cover_seen_stages()
    test_class_incremental_ar_permuted_mapping_is_stable_across_data_seeds()
    test_class_incremental_ar_final_seen_random_baseline()
    test_class_incremental_ar_experiment_imports_configs()
    test_class_incremental_ar_horizon_experiment_imports_configs()
