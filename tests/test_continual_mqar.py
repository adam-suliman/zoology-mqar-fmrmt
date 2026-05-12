import torch

from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.incremental_mqar import IncrementalMQARConfig
from zoology.experiments.rmt_mqar._common import transformer_model
import zoology.train as train_module


def tiny_stage_config(stage_idx: int, num_examples: int = 6):
    return IncrementalMQARConfig(
        stage_idx=stage_idx,
        num_stages=2,
        keys_per_stage=16,
        values_per_stage=16,
        num_kv_pairs=4,
        input_seq_len=32,
        vocab_size=128,
        num_examples=num_examples,
        random_non_queries=False,
        include_slices=True,
    )


def tiny_continual_config(
    max_epochs: int = 1,
    num_examples: int = 2,
    evaluate_future_stages: bool = False,
):
    return ContinualTrainConfig(
        data=ContinualDataConfig(
            train_stage_configs=[
                tiny_stage_config(stage_idx=0, num_examples=num_examples),
                tiny_stage_config(stage_idx=1, num_examples=num_examples),
            ],
            test_stage_configs=[
                tiny_stage_config(stage_idx=0, num_examples=num_examples),
                tiny_stage_config(stage_idx=1, num_examples=num_examples),
            ],
            batch_size=2,
        ),
        model=transformer_model(
            input_seq_len=32,
            d_model=16,
            n_layers=1,
            n_heads=1,
            vocab_size=128,
        ),
        logger=LoggerConfig(backend="none"),
        max_epochs=max_epochs,
        learning_rate=1e-3,
        early_stopping_metric=None,
        evaluate_future_stages=evaluate_future_stages,
        slice_keys=["stage_idx"],
        run_id="continual-smoke",
    )


def test_incremental_mqar_shapes_and_stage_ranges():
    config = tiny_stage_config(stage_idx=1)
    data = config.build(seed=123)
    key_start, key_end = config.key_range()
    value_start, value_end = config.value_range()

    assert data.inputs.shape == (6, 32)
    assert data.labels.shape == (6, 32)
    assert data.slices["stage_idx"] == 1
    assert data.slices["key_start"] == key_start
    assert data.slices["value_start"] == value_start

    context = data.inputs[:, : config.num_kv_pairs * 2]
    keys = context[:, 0::2]
    values = context[:, 1::2]
    assert torch.all((key_start <= keys) & (keys < key_end))
    assert torch.all((value_start <= values) & (values < value_end))

    labels = data.labels[data.labels != -100]
    assert labels.numel() > 0
    assert torch.all((value_start <= labels) & (labels < value_end))


def test_incremental_mqar_stage_ranges_are_disjoint():
    stage0 = tiny_stage_config(stage_idx=0)
    stage1 = tiny_stage_config(stage_idx=1)

    assert stage0.key_range()[1] <= stage1.key_range()[0]
    assert stage0.value_range()[1] <= stage1.value_range()[0]


def test_continual_train_config_imports_and_builds():
    config = tiny_continual_config()

    assert config.training_mode == "continual"
    assert config.evaluate_future_stages is False
    assert len(config.data.train_stage_configs) == 2
    assert len(config.data.test_stage_configs) == 2


def test_continual_eval_summary_tracks_plasticity_bwt_and_forgetting():
    best_stage_accuracy = {}
    learning_stage_accuracy = {}

    first = train_module._summarize_continual_eval(
        stage_idx=0,
        stage_metrics={
            0: {
                "continual/stage_0/accuracy": 0.8,
                "continual/stage_0/loss": 1.0,
            }
        },
        best_stage_accuracy=best_stage_accuracy,
        learning_stage_accuracy={0: 0.8},
        pre_learning_stage_accuracy={0: 0.03},
        pretrain_stage_accuracy={0: 0.03},
        stage_random_accuracy={0: 0.03},
    )
    learning_stage_accuracy[0] = first["continual/current_stage_accuracy"]
    second = train_module._summarize_continual_eval(
        stage_idx=1,
        stage_metrics={
            0: {
                "continual/stage_0/accuracy": 0.5,
                "continual/stage_0/loss": 1.2,
            },
            1: {
                "continual/stage_1/accuracy": 0.7,
                "continual/stage_1/loss": 0.9,
            },
        },
        best_stage_accuracy=best_stage_accuracy,
        learning_stage_accuracy={0: 0.8, 1: 0.7},
        pre_learning_stage_accuracy={0: 0.03, 1: 0.04},
        pretrain_stage_accuracy={0: 0.03, 1: 0.03},
        stage_random_accuracy={0: 0.03, 1: 0.03},
    )

    assert first["continual/seen_avg_accuracy"] == 0.8
    assert first["continual/plasticity"] == 0.8
    assert first["continual/stage_0/fwt_from_random"] == 0.0
    assert second["continual/seen_avg_accuracy"] == 0.6
    assert second["continual/plasticity"] == 0.7
    assert abs(second["continual/stage_0/bwt"] + 0.3) < 1e-8
    assert abs(second["continual/avg_bwt"] + 0.3) < 1e-8
    assert abs(second["continual/stage_1/fwt_from_random"] - 0.01) < 1e-8
    assert abs(second["continual/stage_1/fwt_from_initial"] - 0.01) < 1e-8
    assert abs(second["continual/stage_0/forgetting"] - 0.3) < 1e-8
    assert abs(second["continual/avg_forgetting"] - 0.15) < 1e-8
    assert abs(second["continual/avg_forgetting_from_learning"] - 0.15) < 1e-8


def test_continual_train_smoke_logs_seen_accuracy_and_forgetting():
    logs = []

    class DummyLogger:
        def __init__(self, config):
            pass

        def log_config(self, config):
            pass

        def log_model(self, model, config):
            pass

        def log(self, metrics):
            logs.append(metrics)

        def finish(self):
            pass

    original_logger = train_module.WandbLogger
    train_module.WandbLogger = DummyLogger
    try:
        train_module.train_continual(tiny_continual_config(max_epochs=1, num_examples=2))
    finally:
        train_module.WandbLogger = original_logger

    assert any("continual/seen_avg_accuracy" in metrics for metrics in logs)
    assert any("continual/avg_forgetting" in metrics for metrics in logs)
    assert any("continual/plasticity" in metrics for metrics in logs)
    assert any("continual/avg_bwt" in metrics for metrics in logs)


def test_continual_train_future_eval_logs_pre_learning_and_fwt():
    logs = []

    class DummyLogger:
        def __init__(self, config):
            pass

        def log_config(self, config):
            pass

        def log_model(self, model, config):
            pass

        def log(self, metrics):
            logs.append(metrics)

        def finish(self):
            pass

    original_logger = train_module.WandbLogger
    train_module.WandbLogger = DummyLogger
    try:
        train_module.train_continual(
            tiny_continual_config(
                max_epochs=1,
                num_examples=2,
                evaluate_future_stages=True,
            )
        )
    finally:
        train_module.WandbLogger = original_logger

    assert any("continual/stage_0/pre_learning_accuracy" in metrics for metrics in logs)
    assert any("continual/stage_1/pre_learning_accuracy" in metrics for metrics in logs)
    assert any("continual/avg_fwt_from_random" in metrics for metrics in logs)


if __name__ == "__main__":
    test_incremental_mqar_shapes_and_stage_ranges()
    test_incremental_mqar_stage_ranges_are_disjoint()
    test_continual_train_config_imports_and_builds()
    test_continual_eval_summary_tracks_plasticity_bwt_and_forgetting()
    test_continual_train_smoke_logs_seen_accuracy_and_forgetting()
    test_continual_train_future_eval_logs_pre_learning_and_fwt()
