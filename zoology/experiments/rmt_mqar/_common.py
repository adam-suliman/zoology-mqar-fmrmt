from zoology.config import DataConfig, LoggerConfig, ModelConfig, ModuleConfig, TrainConfig


# MQAR requires vocab_size > input_seq_len. Keep this small, but large enough
# for the default 512-token context/capacity points.
VOCAB_SIZE = 1024


def identity_state_mixer():
    return ModuleConfig(name="torch.nn.Identity", kwargs={})


def transformer_model(
    input_seq_len: int,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    dropout: float = 0.0,
    vocab_size: int = VOCAB_SIZE,
):
    return ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_position_embeddings=input_seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"num_heads": n_heads, "dropout": dropout},
        ),
        name="attention",
    )


def base_rmt_model(
    input_seq_len: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    n_mem: int = 4,
    segment_len: int = 64,
    dropout: float = 0.0,
    detach_memory_between_segments: bool = True,
    memory_layout: str = "read_write",
    vocab_size: int = VOCAB_SIZE,
):
    return ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=1,
        vocab_size=vocab_size,
        max_position_embeddings=input_seq_len,
        state_mixer=identity_state_mixer(),
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.rmt.BaseRMTMixer",
            kwargs={
                "n_heads": n_heads,
                "n_layers": n_layers,
                "n_mem": n_mem,
                "segment_len": segment_len,
                "dropout": dropout,
                "detach_memory_between_segments": detach_memory_between_segments,
                "memory_layout": memory_layout,
            },
        ),
        name="base_rmt",
    )


def fast_memory_rmt_model(
    input_seq_len: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    n_mem: int = 4,
    segment_len: int = 64,
    dropout: float = 0.0,
    detach_memory_between_segments: bool = False,
    memory_layout: str = "read_write",
    rmt_fast_lr: float = 0.05,
    rmt_slow_update_freq: int = 1,
    rmt_clip_memory_grad: float = 1.0,
    reset_memory_each_batch: bool = True,
    reset_memory_each_epoch: bool = False,
    vocab_size: int = VOCAB_SIZE,
):
    return ModelConfig(
        block_type="TransformerBlock",
        d_model=d_model,
        n_layers=1,
        vocab_size=vocab_size,
        max_position_embeddings=input_seq_len,
        state_mixer=identity_state_mixer(),
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.rmt.FastMemoryRMTMixer",
            kwargs={
                "n_heads": n_heads,
                "n_layers": n_layers,
                "n_mem": n_mem,
                "segment_len": segment_len,
                "dropout": dropout,
                "detach_memory_between_segments": detach_memory_between_segments,
                "memory_layout": memory_layout,
                "rmt_fast_lr": rmt_fast_lr,
                "rmt_slow_update_freq": rmt_slow_update_freq,
                "rmt_clip_memory_grad": rmt_clip_memory_grad,
                "reset_memory_each_batch": reset_memory_each_batch,
                "reset_memory_each_epoch": reset_memory_each_epoch,
            },
        ),
        name="fast_memory_rmt",
    )


def comparison_models(input_seq_len: int, segment_len: int = 64, vocab_size: int = VOCAB_SIZE):
    return [
        transformer_model(input_seq_len=input_seq_len, vocab_size=vocab_size),
        base_rmt_model(
            input_seq_len=input_seq_len,
            segment_len=segment_len,
            vocab_size=vocab_size,
        ),
        fast_memory_rmt_model(
            input_seq_len=input_seq_len,
            segment_len=segment_len,
            vocab_size=vocab_size,
        ),
    ]


def class_incremental_models(
    input_seq_len: int,
    segment_len: int = 64,
    vocab_size: int = VOCAB_SIZE,
):
    return [
        transformer_model(
            input_seq_len=input_seq_len,
            vocab_size=vocab_size,
        ),
        base_rmt_model(
            input_seq_len=input_seq_len,
            segment_len=segment_len,
            n_mem=4,
            vocab_size=vocab_size,
        ).model_copy(update={"name": "base_rmt_nmem4"}),
        fast_memory_rmt_model(
            input_seq_len=input_seq_len,
            segment_len=segment_len,
            n_mem=8,
            rmt_fast_lr=0.005,
            rmt_slow_update_freq=4,
            reset_memory_each_batch=False,
            reset_memory_each_epoch=True,
            rmt_clip_memory_grad=1.0,
            vocab_size=vocab_size,
        ).model_copy(update={"name": "fmrmt_stable_lr0p005_slow4"}),
    ]


def train_config(
    model: ModelConfig,
    train_configs,
    test_configs,
    run_id: str,
    max_epochs: int = 4,
    learning_rate: float = 3e-3,
    batch_size=32,
    slice_keys=None,
):
    return TrainConfig(
        data=DataConfig(
            train_configs=train_configs,
            test_configs=test_configs,
            batch_size=batch_size,
        ),
        model=model,
        logger=LoggerConfig(tags=["rmt_mqar"]),
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        early_stopping_metric=None,
        slice_keys=slice_keys or ["num_kv_pairs", "input_seq_len"],
        run_id=run_id,
    )
