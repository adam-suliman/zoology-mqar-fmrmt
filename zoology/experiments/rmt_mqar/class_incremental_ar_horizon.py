from zoology.config import ContinualDataConfig, ContinualTrainConfig, LoggerConfig
from zoology.data.class_incremental_ar import ClassIncrementalARConfig
from zoology.experiments.rmt_mqar._common import class_incremental_models


SEEDS = [123, 456, 789]
HORIZON_POINTS = [5, 10, 20]
VOCAB_SIZE = 1024
ASSOCIATIONS_PER_STAGE = 16
NUM_QUERY_ASSOCIATIONS = 8
INPUT_SEQ_LEN = 128
TRAIN_EXAMPLES = 2_048
TEST_EXAMPLES = 512
MAX_EPOCHS = 8
BATCH_SIZE = 64
SEGMENT_LEN = 64


def stage_config(
    stage_idx: int,
    num_stages: int,
    num_examples: int,
    eval_mode: str,
    distractor_mode: str,
):
    return ClassIncrementalARConfig(
        stage_idx=stage_idx,
        num_stages=num_stages,
        associations_per_stage=ASSOCIATIONS_PER_STAGE,
        num_query_associations=NUM_QUERY_ASSOCIATIONS,
        input_seq_len=INPUT_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_examples=num_examples,
        eval_mode=eval_mode,
        distractor_mode=distractor_mode,
        include_slices=True,
    )


def data_config(num_stages: int, seed: int):
    return ContinualDataConfig(
        train_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=TRAIN_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        test_stage_configs=[
            stage_config(
                stage_idx=stage_idx,
                num_stages=num_stages,
                num_examples=TEST_EXAMPLES,
                eval_mode="current",
                distractor_mode="current",
            )
            for stage_idx in range(num_stages)
        ],
        batch_size=BATCH_SIZE,
        seed=seed,
    )


configs = [
    ContinualTrainConfig(
        data=data_config(num_stages=num_stages, seed=seed),
        model=model,
        logger=LoggerConfig(tags=["rmt_mqar", "class_incremental_ar", "horizon"]),
        max_epochs=MAX_EPOCHS,
        learning_rate=3e-3,
        early_stopping_metric=None,
        evaluate_future_stages=True,
        slice_keys=[
            "stage_idx",
            "eval_mode",
            "associations_per_stage",
            "num_query_associations",
            "input_seq_len",
        ],
        seed=seed,
        run_id=f"{model.name}-class-incremental-ar-{num_stages}stage-seed{seed}",
    )
    for num_stages in HORIZON_POINTS
    for seed in SEEDS
    for model in class_incremental_models(
        input_seq_len=INPUT_SEQ_LEN,
        segment_len=SEGMENT_LEN,
        vocab_size=VOCAB_SIZE,
    )
]
