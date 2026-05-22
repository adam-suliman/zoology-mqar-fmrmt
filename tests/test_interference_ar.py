import importlib.util

import torch

from zoology.config import ContinualTrainConfig
from zoology.data.interference_ar import InterferenceARConfig


def tiny_config(
    target_policy: str = "latest",
    num_interference_pairs: int = 2,
    num_query_associations: int = 4,
    update_value_mode: str = "random",
):
    return InterferenceARConfig(
        stage_idx=1,
        num_stages=3,
        associations_per_stage=4,
        num_query_associations=num_query_associations,
        num_interference_pairs=num_interference_pairs,
        input_seq_len=40,
        vocab_size=128,
        num_examples=8,
        eval_mode="current",
        distractor_mode="current",
        target_policy=target_policy,
        update_value_mode=update_value_mode,
        include_slices=True,
    )


def non_ignored_labels(data):
    return data.labels[data.labels != -100]


def test_interference_ar_shapes_and_label_range():
    config = tiny_config(target_policy="latest")
    data = config.build(seed=123)
    labels = non_ignored_labels(data)
    value_start = config.vocab_size // 2 + config.stage_idx * config.associations_per_stage
    value_end = value_start + config.associations_per_stage

    assert data.inputs.shape == (8, 40)
    assert data.labels.shape == (8, 40)
    assert labels.numel() > 0
    assert torch.all((value_start <= labels) & (labels < value_end))
    assert data.slices["num_interference_pairs"] == 2
    assert data.slices["target_policy"] == "latest"


def test_interference_ar_old_and_latest_targets_differ():
    old = tiny_config(target_policy="old").build(seed=123)
    latest = tiny_config(target_policy="latest").build(seed=123)

    old_labels = old.labels[old.labels != -100]
    latest_labels = latest.labels[latest.labels != -100]
    assert old_labels.shape == latest_labels.shape
    assert torch.any(old_labels != latest_labels)


def test_interference_ar_fixed_shift_latest_targets_follow_update_mapping():
    config = tiny_config(
        target_policy="latest",
        num_query_associations=4,
        num_interference_pairs=4,
        update_value_mode="fixed_shift",
    )
    data = config.build(seed=123)

    label_mask = data.labels != -100
    query_keys = data.inputs[label_mask]
    labels = data.labels[label_mask]
    value_start = config.vocab_size // 2 + config.stage_idx * config.associations_per_stage
    key_start = 1 + config.stage_idx * config.associations_per_stage
    expected = value_start + ((query_keys - key_start + 1) % config.associations_per_stage)

    assert torch.equal(labels, expected)
    assert data.slices["update_value_mode"] == "fixed_shift"


def test_interference_ar_no_conflict_imports():
    config = tiny_config(target_policy="latest", num_interference_pairs=0)
    data = config.build(seed=123)

    assert data.slices["num_interference_pairs"] == 0
    assert data.slices["random_accuracy"] == config.random_accuracy()


def test_class_incremental_interference_experiment_imports_configs():
    spec = importlib.util.spec_from_file_location(
        "class_incremental_ar_interference",
        "zoology/experiments/rmt_mqar/class_incremental_ar_interference.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module.configs) == 27
    assert all(isinstance(config, ContinualTrainConfig) for config in module.configs)
    assert all(config.evaluate_future_stages for config in module.configs)


if __name__ == "__main__":
    test_interference_ar_shapes_and_label_range()
    test_interference_ar_old_and_latest_targets_differ()
    test_interference_ar_no_conflict_imports()
    test_interference_ar_fixed_shift_latest_targets_follow_update_mapping()
    test_class_incremental_interference_experiment_imports_configs()
