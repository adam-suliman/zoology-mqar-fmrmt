from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rmt_mqar._common import (
    VOCAB_SIZE,
    comparison_models,
    train_config,
)


# Add 1024 here for a larger context run after the small defaults are validated.
CONTEXT_LENGTHS = [64, 128, 256, 512]
NUM_KV_PAIRS = 8

configs = []

for input_seq_len in CONTEXT_LENGTHS:
    train_configs = [
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=input_seq_len,
            num_examples=512,
            num_kv_pairs=NUM_KV_PAIRS,
        )
    ]
    test_configs = [
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=input_seq_len,
            num_examples=128,
            num_kv_pairs=NUM_KV_PAIRS,
        )
    ]

    for model in comparison_models(input_seq_len=input_seq_len, segment_len=64):
        configs.append(
            train_config(
                model=model,
                train_configs=train_configs,
                test_configs=test_configs,
                run_id=f"{model.name}-context{input_seq_len}",
                max_epochs=4,
                learning_rate=3e-3,
                batch_size=32,
            )
        )

