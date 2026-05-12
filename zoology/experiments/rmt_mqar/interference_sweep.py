from zoology.data.forgetting_mqar import ForgettingMQARConfig
from zoology.experiments.rmt_mqar._common import (
    VOCAB_SIZE,
    comparison_models,
    train_config,
)


# Existing ForgettingMQARConfig covers repeated keys with changed values and
# queries the latest value. TODO: extend zoology/data/forgetting_mqar.py or add a
# sibling DataSegmentConfig to support near-duplicate keys and a target_policy
# option for querying either the old value or the latest value.
INTERFERENCE_POINTS = [
    {"name": "few_updates", "num_kv_pairs": 8, "num_updates": 2, "input_seq_len": 128},
    {"name": "many_updates", "num_kv_pairs": 16, "num_updates": 8, "input_seq_len": 256},
]

configs = []

for point in INTERFERENCE_POINTS:
    train_configs = [
        ForgettingMQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=point["input_seq_len"],
            num_examples=512,
            num_kv_pairs=point["num_kv_pairs"],
            num_updates=point["num_updates"],
        )
    ]
    test_configs = [
        ForgettingMQARConfig(
            vocab_size=VOCAB_SIZE,
            input_seq_len=point["input_seq_len"],
            num_examples=128,
            num_kv_pairs=point["num_kv_pairs"],
            num_updates=point["num_updates"],
        )
    ]

    for model in comparison_models(input_seq_len=point["input_seq_len"], segment_len=64):
        configs.append(
            train_config(
                model=model,
                train_configs=train_configs,
                test_configs=test_configs,
                run_id=f"{model.name}-{point['name']}",
                max_epochs=4,
                learning_rate=3e-3,
                batch_size=32,
                slice_keys=["num_kv_pairs", "num_updates", "input_seq_len"],
            )
        )

