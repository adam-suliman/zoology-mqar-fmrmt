# RMT MQAR Results

This file tracks the important local results for the RMT/MQAR work. The raw
artifacts live in `results/`; this report is the hand-curated summary we can
update after each meaningful run.

Note: class-incremental AR results before 2026-05-13 used cumulative-prefix test
splits. Current formal CL runs use stage-local test splits so plasticity, BWT,
and forgetting are not conflated. Treat the older cumulative-prefix tables as
historical architecture probes, not directly comparable formal CL metrics.

## Metric Notes

- `final cumulative accuracy`: accuracy on the final class-incremental test
  stage, where all seen answer/value classes are in the candidate distribution.
- `seen avg accuracy`: mean accuracy across all seen stages after the final
  training stage.
- `avg forgetting`: mean drop from each stage's best historical accuracy to its
  final accuracy.
- `plasticity`: current-stage accuracy immediately after training that stage.
- `BWT`: final old-stage accuracy minus the accuracy immediately after that
  stage was learned; negative values indicate forgetting.
- For class-incremental AR with 5 stages and 16 values per stage, final random
  cumulative-prefix accuracy is `1 / (5 * 16) = 0.0125`; stage-local random
  accuracy is `1 / 16 = 0.0625`.

## Formal Class-Incremental AR CL Metrics

Source: `results/class_incremental_ar_formal_cl_20260513_022211.json`

Setup:

- Task: class-incremental associative retrieval with stage-local test splits.
- Stages: 5.
- New associations per stage: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples per stage: 4096.
- Test examples per stage: 1024.
- Epochs per stage: 16.
- Seeds: 123, 456, 789.
- Segment length for RMT variants: 64.
- `evaluate_future_stages=True`.
- Stage-local random accuracy: 0.0625.

| Model | Final Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMRMT n_mem=8, lr=0.005, slow_freq=4, epoch reset | 0.8501 +/- 0.0488 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1874 +/- 0.0609 | 0.1499 +/- 0.0488 |
| Base RMT n_mem=4 | 0.8081 +/- 0.0147 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2399 +/- 0.0183 | 0.1919 +/- 0.0147 |
| Transformer/MHA | 0.1321 +/- 0.1024 | 0.8925 +/- 0.0713 | 0.4832 +/- 0.3324 | -0.9504 +/- 0.0455 | 0.7604 +/- 0.0364 |

Final stage-local accuracies after all training:

| Model | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.3011 | 0.9997 | 0.9495 | 1.0000 | 1.0000 |
| Base RMT n_mem=4 | 0.0416 | 1.0000 | 0.9987 | 1.0000 | 1.0000 |
| Transformer/MHA | 0.0000 | 0.0000 | 0.0148 | 0.1627 | 0.4832 |

Learning-time stage accuracies:

| Model | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Base RMT n_mem=4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Transformer/MHA | 1.0000 | 1.0000 | 0.9851 | 0.9941 | 0.4832 |

Interpretation:

- Base RMT and FMRMT show no loss of plasticity in this 5-stage setting: each
  newly introduced stage reaches 1.0 accuracy immediately after training.
- Their difference is retention. FMRMT has higher final seen accuracy and less
  negative BWT than Base RMT, mostly because it retains stage 0 better.
- Transformer/MHA shows both severe forgetting and emerging plasticity failure:
  it learns early stages initially, then forgets them, and its final-stage
  plasticity drops sharply.
- FWT is essentially zero or below random for all models, so this setup does not
  show positive forward transfer.

## Historical Cumulative-Prefix Class-Incremental Result

Source: `results/best_arch_class_incremental_ar_20260511_200731.json`

Setup:

- Task: class-incremental associative retrieval.
- Stages: 5.
- New associations per stage: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples per stage: 4096.
- Test examples per stage: 1024.
- Epochs per stage: 16.
- Seeds: 123, 456, 789, 101112, 131415.
- Segment length for RMT variants: 64.

| Model | Final Cumulative Accuracy | Seen Avg Accuracy | Avg Forgetting | Notes |
| --- | ---: | ---: | ---: | --- |
| FMRMT n_mem=8, lr=0.005, slow_freq=4, epoch reset | 0.8584 +/- 0.0435 | 0.6620 +/- 0.0878 | 0.1833 +/- 0.0638 | Best current default; strongest stability. |
| FMRMT n_mem=8, lr=0.01, slow_freq=4, epoch reset | 0.8365 +/- 0.1006 | 0.6423 +/- 0.1639 | 0.2044 +/- 0.1013 | Highest single run, but more seed variance. |
| Base RMT n_mem=4 | 0.7959 +/- 0.0263 | 0.5483 +/- 0.0323 | 0.2042 +/- 0.0082 | Strong stable baseline. |
| Transformer/MHA | 0.1237 +/- 0.0855 | 0.0393 +/- 0.0247 | 0.3930 +/- 0.0138 | Mostly fails continual retention. |

Per-seed final cumulative accuracy:

| Model | 123 | 456 | 789 | 101112 | 131415 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMRMT lr=0.005 slow_freq=4 | 0.8938 | 0.8879 | 0.8430 | 0.7891 | 0.8783 |
| FMRMT lr=0.01 slow_freq=4 | 0.8785 | 0.8623 | 0.9617 | 0.7832 | 0.6967 |
| Base RMT n_mem=4 | 0.7977 | 0.8287 | 0.8009 | 0.7970 | 0.7551 |
| Transformer/MHA | 0.0533 | 0.2273 | 0.0409 | 0.2009 | 0.0961 |

Interpretation:

- The stable FMRMT setting is the best current candidate: it improves over Base
  RMT by about 6.25 final accuracy points and reduces average forgetting.
- The lr=0.01 FMRMT setting is useful but less reliable. It produced the best
  single run, but also the weakest tuned-FMRMT seed.
- Base RMT is a meaningful baseline. It is much stronger than Transformer/MHA
  and has low variance.
- Transformer/MHA can occasionally learn some final-stage associations, but it
  does not reliably retain earlier stages in this setup.

## FMRMT Tuning Sweep

Source: `results/fmrmt_tuning_class_incremental_ar_20260511_193201.json`

Setup:

- Same 5-stage class-incremental AR task as above.
- Seeds: 123, 456, 789.
- FMRMT: `n_mem=8`, `reset_memory_each_batch=False`,
  `reset_memory_each_epoch=True`, `rmt_clip_memory_grad=1.0`.
- Swept `rmt_fast_lr in {0.005, 0.01, 0.02, 0.05}` and
  `rmt_slow_update_freq in {1, 4}`.
- Baselines: Base RMT with `n_mem=4` and `n_mem=8`.

| Model | Final Cumulative Accuracy | Seen Avg Accuracy | Avg Forgetting | Notes |
| --- | ---: | ---: | ---: | --- |
| FMRMT lr=0.01, slow_freq=4 | 0.9008 +/- 0.0533 | 0.7459 +/- 0.1096 | 0.1357 +/- 0.0472 | Best 3-seed tuning mean. |
| FMRMT lr=0.005, slow_freq=4 | 0.8749 +/- 0.0278 | 0.6937 +/- 0.0687 | 0.1596 +/- 0.0582 | More stable; became current default after 5-seed check. |
| FMRMT lr=0.02, slow_freq=4 | 0.8518 +/- 0.1208 | 0.6683 +/- 0.2456 | 0.1846 +/- 0.1366 | Strong but high variance. |
| FMRMT lr=0.05, slow_freq=4 | 0.8190 +/- 0.0959 | 0.6378 +/- 0.2023 | 0.1725 +/- 0.1048 | Still above Base RMT mean in this 3-seed run. |
| Base RMT n_mem=4 | 0.8091 +/- 0.0171 | 0.5626 +/- 0.0333 | 0.2036 +/- 0.0104 | Strong 3-seed baseline. |
| Base RMT n_mem=8 | 0.7281 +/- 0.0354 | 0.4641 +/- 0.0483 | 0.2236 +/- 0.0162 | More memory tokens did not help Base RMT here. |

Key finding:

- Slow model updates every 4 batches plus epoch-level fast-memory reset were the
  useful changes. Fast-memory updates with `slow_freq=1` generally behaved much
  closer to Base RMT and did not unlock the same retention gains.

## Initial Class-Incremental AR Run

Source: `results/class_incremental_ar_end_to_end_20260511_173007.json`

Setup:

- 5 stages, 16 associations per stage, 8 queried associations per sequence.
- Sequence length 128, segment length 64.
- 4096 train examples and 1024 test examples per stage.
- 16 epochs per stage.
- Seeds: 123, 456, 789.

| Model | Final Cumulative Accuracy | Seen Avg Accuracy | Avg Forgetting |
| --- | ---: | ---: | ---: |
| Base RMT | 0.8091 +/- 0.0171 | 0.5626 +/- 0.0333 | 0.2036 +/- 0.0104 |
| Fast Memory RMT default | 0.6848 +/- 0.0802 | 0.4191 +/- 0.1155 | 0.2337 +/- 0.0562 |
| Transformer/MHA | 0.1072 +/- 0.1042 | 0.0307 +/- 0.0301 | 0.3984 +/- 0.0135 |

Interpretation:

- Base RMT was already strong before tuning.
- The first FMRMT defaults were not good enough; they underperformed Base RMT.
- This motivated the later fast-memory tuning sweep.

## Vocab-Growth MQAR Diagnostics

Source: `results/vocab_growth_diagnostics_20260511_203603.json`

Setup:

- Task: stage-local episodic MQAR with disjoint key/value token ranges per
  stage.
- Diagnostics: 1-stage solvability, 2-stage forgetting, and 4-stage full run.
- `keys_per_stage=64`, `values_per_stage=32`, `num_kv_pairs=8`.
- Sequence length 128, segment length 64.
- 4096 train examples and 1024 test examples per stage.
- 16 epochs per stage, 3 seeds.
- Random stage accuracy: `1 / 32 = 0.03125`.
- Models: Transformer/MHA, Base RMT n_mem=4, and the current stable FMRMT
  default from class-incremental AR.

1-stage solvability:

| Model | Final Accuracy | Train Loss | Notes |
| --- | ---: | ---: | --- |
| Base RMT n_mem=4 | 0.1294 +/- 0.0113 | 2.8256 +/- 0.3205 | Best, but still weak. |
| Transformer/MHA | 0.1171 +/- 0.0104 | 3.1202 +/- 0.1879 | Above random, not solved. |
| FMRMT stable lr=0.005 slow_freq=4 | 0.0313 +/- 0.0029 | 3.4426 +/- 0.0251 | Essentially random. |

2-stage forgetting:

| Model | Final Seen Avg Accuracy | Final Stage 0 Accuracy | Final Stage 1 Accuracy | Avg Forgetting |
| --- | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0360 +/- 0.0204 | 0.0393 | 0.0327 | 0.0432 +/- 0.0193 |
| Base RMT n_mem=4 | 0.0310 +/- 0.0219 | 0.0249 | 0.0371 | 0.0562 +/- 0.0203 |
| FMRMT stable lr=0.005 slow_freq=4 | 0.0192 +/- 0.0019 | 0.0069 | 0.0315 | 0.0135 +/- 0.0044 |

4-stage full run:

| Model | Final Seen Avg Accuracy | Final Stage Accuracies 0/1/2/3 | Avg Forgetting |
| --- | ---: | --- | ---: |
| Transformer/MHA | 0.0385 +/- 0.0494 | 0.0271 / 0.0238 / 0.0424 / 0.0607 | 0.0441 +/- 0.0076 |
| Base RMT n_mem=4 | 0.0293 +/- 0.0360 | 0.0028 / 0.0308 / 0.0438 / 0.0400 | 0.0593 +/- 0.0149 |
| FMRMT stable lr=0.005 slow_freq=4 | 0.0103 +/- 0.0044 | 0.0001 / 0.0001 / 0.0098 / 0.0312 | 0.0254 +/- 0.0068 |

Historical 4-stage source:
`results/continual_mqar_full_reasonable_20260511_143117.json`

| Model | Final Seen Avg Accuracy | Avg Forgetting |
| --- | ---: | ---: |
| Transformer/MHA | 0.0385 +/- 0.0494 | 0.0441 +/- 0.0076 |
| Fast Memory RMT default | 0.0367 +/- 0.0275 | 0.0738 +/- 0.0160 |
| Base RMT | 0.0293 +/- 0.0360 | 0.0593 +/- 0.0149 |

Interpretation:

- Vocab-growth MQAR is not currently a clean architecture-ranking benchmark.
  The 1-stage check is only partially learnable for Transformer/Base RMT and is
  random for the stable FMRMT setting.
- The 2-stage and 4-stage checks mostly measure task collapse rather than
  controlled forgetting. Old-stage accuracy often goes to zero, while current
  stage accuracy is also near random.
- The FMRMT setting that works best on class-incremental AR does not transfer to
  this episodic vocab-growth setup. Its slow update cadence and epoch-reset fast
  memory are likely too conservative for this task.
- Class-incremental AR remains the main benchmark. Vocab-growth should be
  redesigned or made easier before using it for model comparison.

## Open Controls

- Run the same best-architecture comparison with `segment_len=128` to check
  whether RMT/FMRMT gains depend on recurrence across segments or mostly on the
  internal RMT block.
- Add a compute-matched Transformer baseline if we want a stronger fairness
  claim than "same Zoology default width/depth".
- Add old-vs-new class accuracy to separate plasticity from retention more
  directly.
- Repeat the stable FMRMT default on harder variants: more stages, more
  associations per stage, and longer distractor spans.
- Redesign vocab-growth MQAR into an easier calibration ladder before using it
  for ranking: fewer query pairs, smaller value ranges, more train examples, or
  fixed per-stage associations instead of fully episodic random values.
