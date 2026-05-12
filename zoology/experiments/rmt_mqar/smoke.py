from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rmt_mqar._common import (
    VOCAB_SIZE,
    comparison_models,
    train_config,
)


INPUT_SEQ_LEN = 64
NUM_KV_PAIRS = 4

train_configs = [
    MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=INPUT_SEQ_LEN,
        num_examples=64,
        num_kv_pairs=NUM_KV_PAIRS,
    )
]

test_configs = [
    MQARConfig(
        vocab_size=VOCAB_SIZE,
        input_seq_len=INPUT_SEQ_LEN,
        num_examples=32,
        num_kv_pairs=NUM_KV_PAIRS,
    )
]

configs = [
    train_config(
        model=model,
        train_configs=train_configs,
        test_configs=test_configs,
        run_id=f"{model.name}-smoke",
        max_epochs=1,
        learning_rate=3e-3,
        batch_size=16,
    )
    for model in comparison_models(input_seq_len=INPUT_SEQ_LEN, segment_len=32)
]

