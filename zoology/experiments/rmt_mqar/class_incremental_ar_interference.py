from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.interference_ar import InterferenceARConfig
from zoology.experiments.rmt_mqar._common import class_incremental_models


SEEDS = [123, 456, 789]
NUM_STAGES = 5
VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
TRAIN_EXAMPLES = 2_048
TEST_EXAMPLES = 512
MAX_EPOCHS = 8
BATCH_SIZE = 64
SEGMENT_LEN = 64


INTERFERENCE_POINTS = [
    {
        "name": "no_conflict_latest",
        "num_interference_pairs": 0,
        "target_policy": "latest",
    },
    {
        "name": "repeated_key_latest",
        "num_interference_pairs": 4,
        "target_policy": "latest",
    },
    {
        "name": "repeated_key_old",
        "num_interference_pairs": 4,
        "target_policy": "old",
    },
]


def stage_config(
    stage_idx: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
    num_interference_pairs: int,
    target_policy: str,
):
    return InterferenceARConfig(
        stage_idx=stage_idx,
        num_stages=NUM_STAGES,
        associations_per_stage=ASSOCIATIONS_PER_STAGE,
        num_query_associations=NUM_QUERY_ASSOCIATIONS,
        num_interference_pairs=num_interference_pairs,
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        target_policy=target_policy,
        include_slices=True,
    )


def data_config(seed: int, point: dict):
    return ContinualDataConfig(
        train_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_examples=TRAIN_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
                num_interference_pairs=point["num_interference_pairs"],
                target_policy=point["target_policy"],
            )
            for stage_idx in range(NUM_STAGES)
        ],
        test_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_examples=TEST_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
                num_interference_pairs=point["num_interference_pairs"],
                target_policy=point["target_policy"],
            )
            for stage_idx in range(NUM_STAGES)
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


configs = [
    ContinualTrainConfig(
        data=data_config(seed=seed, point=point),
        model=model,
        logger=LoggerConfig(tags=["rmt_mqar", "class_incremental_ar", "interference"]),
        max_epochs=MAX_EPOCHS,
        learning_rate=3e-3,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        slice_keys=[
            "stage_idx",
            "eval_mode",
            "associations_per_stage",
            "num_query_associations",
            "num_interference_pairs",
            "target_policy",
            "input_seq_len",
        ],
        seed=seed,
        run_id=f"{model.name}-{point['name']}-seed{seed}",
    )
    for point in INTERFERENCE_POINTS
    for seed in SEEDS
    for model in class_incremental_models(
        input_seq_len=INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        vocab_size=VOCAB_SIZE,
    )
]
