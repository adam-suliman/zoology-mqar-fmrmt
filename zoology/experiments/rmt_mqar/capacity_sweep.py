from zoology.data.multiquery_ar import MQARConfig
from zoology.experiments.rmt_mqar._common import (
    VOCAB_SIZE,
    comparison_models,
    train_config,
)


# Keep defaults small for launch checks. Increase num_examples/max_epochs for real sweeps.
CAPACITY_POINTS = [
    {"num_kv_pairs": 8, "input_seq_len": 64},
    {"num_kv_pairs": 16, "input_seq_len": 128},
    {"num_kv_pairs": 32, "input_seq_len": 256},
    {"num_kv_pairs": 64, "input_seq_len": 512},
]

configs = []

for point in CAPACITY_POINTS:
    num_kv_pairs = point["num_kv_pairs"]
    input_seq_len = point["input_seq_len"]
    train_configs = [
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=input_seq_len,
            num_examples=512,
            num_kv_pairs=num_kv_pairs,
        )
    ]
    test_configs = [
        MQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=input_seq_len,
            num_examples=128,
            num_kv_pairs=num_kv_pairs,
        )
    ]

    for model in comparison_models(input_seq_len=input_seq_len, segment_len=64):
        configs.append(
            train_config(
                model=model,
                train_configs=train_configs,
                test_configs=test_configs,
                run_id=f"{model.name}-kv{num_kv_pairs}-seq{input_seq_len}",
                max_epochs=4,
                learning_rate=3e-3,
                batch_size=32,
            )
        )

