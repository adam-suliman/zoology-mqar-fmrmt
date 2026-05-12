from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.incremental_mqar import IncrementalMQARConfig
from zoology.experiments.rmt_mqar._common import comparison_models


NUM_STAGES = 4
VOCAB_SIZE = 2048
KEYS_PER_STAGE = 128
VALUES_PER_STAGE = 128
NUM_KV_PAIRS = 8
INPUT_SEQ_LEN = 128
TRAIN_EXAMPLES = 1_000
TEST_EXAMPLES = 256
MAX_EPOCHS = 4
BATCH_SIZE = 32
SEGMENT_LEN = 64


def stage_config(stage_idx: int, num_examples: int):
    return IncrementalMQARConfig(
        stage_idx=stage_idx,
        num_stages=NUM_STAGES,
        keys_per_stage=KEYS_PER_STAGE,
        values_per_stage=VALUES_PER_STAGE,
        num_kv_pairs=NUM_KV_PAIRS,
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        random_non_queries=True,
        include_slices=True,
    )


data = ContinualDataConfig(
    train_stage_configs=[
        stage_config(stage_idx=stage_idx, num_examples=TRAIN_EXAMPLES)
        for stage_idx in range(NUM_STAGES)
    ],
    test_stage_configs=[
        stage_config(stage_idx=stage_idx, num_examples=TEST_EXAMPLES)
        for stage_idx in range(NUM_STAGES)
    ],
    batch_size=BATCH_SIZE,
)


configs = [
    ContinualTrainConfig(
        data=data,
        model=model,
        logger=LoggerConfig(tags=["rmt_mqar", "continual_vocab"]),
        max_epochs=MAX_EPOCHS,
        learning_rate=3e-3,
        early_stopping_metric=None,
        slice_keys=["stage_idx", "num_kv_pairs", "input_seq_len"],
        run_id=f"{model.name}-continual-vocab",
    )
    for model in comparison_models(
        input_seq_len=INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        vocab_size=VOCAB_SIZE,
    )
]
