from zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity import (
    ExperimentArgs,
    cumulative_train_config,
    current_test_config,
    default_vocab_size,
    detect_first_plasticity_loss_stage,
    parse_args,
)


def tiny_args(**kwargs):
    values = {
        "num_stages": 20,
        "seeds": [123],
        "examples_per_seen_stage": 256,
        "test_examples_per_stage": 128,
        "epochs_per_stage": 2,
        "output_prefix": "test",
    }
    values.update(kwargs)
    return ExperimentArgs(**values)


def test_default_vocab_size_matches_stage_capacity():
    assert default_vocab_size(20, 16) == 1024
    assert default_vocab_size(50, 16) == 2048


def test_parse_rejects_oversized_stage_count_for_explicit_vocab():
    try:
        parse_args([
            "--num-stages", "50",
            "--seeds", "123",
            "--examples-per-seen-stage", "256",
            "--test-examples-per-stage", "128",
            "--epochs-per-stage", "2",
            "--output-prefix", "test",
            "--vocab-size", "1024",
        ])
    except ValueError as exc:
        assert "does not fit" in str(exc)
    else:
        raise AssertionError("Expected 50 stages to require a larger vocab")


def test_cumulative_train_config_uses_seen_seen_and_scaled_examples():
    args = tiny_args()
    config = cumulative_train_config(args, stage_idx=4)

    assert config.eval_mode == "seen"
    assert config.distractor_mode == "seen"
    assert config.num_examples == 256 * 5
    assert config.vocab_size == 1024
    assert config.value_mapping == "permuted"


def test_current_probe_uses_current_seen():
    args = tiny_args(num_stages=50)
    config = current_test_config(args, stage_idx=10)

    assert config.eval_mode == "current"
    assert config.distractor_mode == "seen"
    assert config.num_examples == 128
    assert config.vocab_size == 2048


def stage(fresh, continuous):
    return {
        "stage_idx": 0,
        "fresh": {"current_final_accuracy": fresh},
        "continuous": {"current_final_accuracy": continuous},
        "plasticity_gap_final": fresh - continuous,
    }


def test_detect_first_plasticity_loss_stage_requires_consecutive_flags():
    stages = [
        {**stage(1.0, 1.0), "stage_idx": 0},
        {**stage(1.0, 0.75), "stage_idx": 1},
        {**stage(1.0, 0.70), "stage_idx": 2},
        {**stage(1.0, 0.95), "stage_idx": 3},
    ]
    detection = detect_first_plasticity_loss_stage(stages)

    assert detection["stage_flags"] == [False, True, True, False]
    assert detection["first_plasticity_loss_stage"] == 1
