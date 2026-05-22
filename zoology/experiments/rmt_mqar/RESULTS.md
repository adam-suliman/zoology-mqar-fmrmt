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
- `stage_train_wall_seconds` and `stage_seen_eval_wall_seconds`: wall-clock time
  for the current stage's training block and seen-stage validation block.
  New continual runs after 2026-05-17 also log processed examples/tokens,
  train/eval batches, optimizer steps, and examples/tokens per second. Older
  artifacts in this file do not include these timing fields and need reruns for
  compute-aware comparison.
- For class-incremental AR with 5 stages and 16 values per stage, final random
  cumulative-prefix accuracy is `1 / (5 * 16) = 0.0125`; stage-local random
  accuracy is `1 / 16 = 0.0625`.

## One-Stage Solvability Calibration

Clean class-incremental source:
`results/class_incremental_ar_one_stage_formal_cl_20260517_025825.json`

Repeated-key latest-value source:
`results/interference_ar_latest_updates4_one_stage_formal_cl_20260517_030058.json`

Clean class-incremental setup:

- Task: class-incremental associative retrieval with exactly 1 stage.
- New associations: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples: 4096.
- Test examples: 1024.
- Epochs: 16.
- Seeds: 123, 456, 789.
- Stage-local random accuracy: 0.0625.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Plasticity | Avg BWT | Forgetting From Learning | Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0027 +/- 0.0000 |
| Base RMT n_mem=4 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0024 +/- 0.0000 |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0106 +/- 0.0002 |

Repeated-key latest-value setup:

- Task: one-stage `InterferenceARConfig` with 4 repeated-key conflicts and
  `target_policy="latest"`.
- New associations: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples: 2048.
- Test examples: 512.
- Epochs: 8.
- Seeds: 123, 456, 789.
- Stage-local random accuracy: 0.0625.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Plasticity | Avg BWT | Forgetting From Learning | Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.4496 +/- 0.0034 | 0.4496 +/- 0.0034 | 0.4496 +/- 0.0034 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.4435 +/- 0.0018 |
| Base RMT n_mem=4 | 0.5046 +/- 0.0066 | 0.5046 +/- 0.0066 | 0.5046 +/- 0.0066 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.7082 +/- 0.1769 |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 2.1526 +/- 0.0309 |

Interpretation:

- The clean class-incremental task passes the necessary single-stage
  calibration: all three current comparison models solve it perfectly across 3
  seeds. Later clean class-incremental plasticity failures are therefore not due
  to basic one-stage unsolvability under the standard formal budget.
- BWT and forgetting are zero in one-stage runs by construction, because there
  are no old stages. The meaningful calibration metrics here are current-stage
  accuracy, learning accuracy, and train loss.
- The repeated-key latest-value variant does not pass single-stage solvability
  under the current lower-budget interference setup. Scores cluster around
  0.45-0.50, far above random but far below perfect.
- Therefore, the repeated-key latest-value formal CL table should be interpreted
  as a mixture of within-stage overwrite difficulty and continual-learning
  effects. Its low current-stage accuracy is real low learning accuracy, but it
  is not clean evidence of continual plasticity loss unless the one-stage
  latest-value task is first made solvable.

## Random-Permuted Stage Mapping Smoke

Source:
`results/class_incremental_ar_permuted_smoke_one_stage_permuted_continual3_permuted_smoke_20260522_132623.json`

Setup:

- Task: class-incremental AR with fixed per-stage key/value tables, but values
  are randomly permuted within each stage instead of offset-aligned.
- Association table seed: 20260522.
- Data seeds: 123, 456.
- New associations per stage: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- One-stage smoke: 1024 train examples, 256 test examples, 8 epochs.
- Three-stage smoke: 512 train examples/stage, 256 test examples/stage,
  4 epochs/stage.
- Stage-local random accuracy: 0.0625.

| Task | Model | Seen Avg Accuracy | Avg Learning Accuracy | Plasticity | Avg BWT | Forgetting From Learning |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 stage | Transformer/MHA | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 1 stage | Base RMT n_mem=4 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 1 stage | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 3-stage smoke | Transformer/MHA | 0.1300 +/- 0.0277 | 0.3647 +/- 0.0089 | 0.0000 +/- 0.0000 | -0.3519 +/- 0.0282 | 0.2346 +/- 0.0188 |
| 3-stage smoke | Base RMT n_mem=4 | 0.7722 +/- 0.0598 | 0.7722 +/- 0.0598 | 0.3167 +/- 0.1794 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| 3-stage smoke | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.5733 +/- 0.0168 | 0.4639 +/- 0.0070 | 0.0059 +/- 0.0059 | 0.1642 +/- 0.0358 | 0.0000 +/- 0.0000 |

Interpretation:

- The permuted table construction passes the key sanity check: all current
  comparison models solve the one-stage variant perfectly across both seeds.
- The per-stage association table is controlled by `association_table_seed`,
  separately from the train/test data seed. This keeps the same fixed mapping
  across train and test while still allowing independent sample draws.
- The three-stage smoke is intentionally cheap and undertrained. It confirms
  that the continual pipeline runs stably with permuted mappings, but it should
  not be used as a formal model comparison.
- Because the values are no longer offset-aligned with keys, this variant is a
  better control for checking whether clean class-incremental AR gains depend on
  the arithmetic key-offset/value-offset shortcut.

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

## Formal Class-Incremental AR Horizon Sweep

Source: `results/class_incremental_ar_horizon_formal_cl_20260513_054644.json`

Setup:

- Task: class-incremental associative retrieval with stage-local test splits.
- Horizon points: 5, 10, and 20 stages.
- New associations per stage: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples per stage: 2048.
- Test examples per stage: 512.
- Epochs per stage: 8.
- Seeds: 123, 456, 789.
- Segment length for RMT variants: 64.
- `evaluate_future_stages=True`.
- Stage-local random accuracy: 0.0625.

| Horizon | Model | Final Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 stages | Transformer/MHA | 0.0380 +/- 0.0166 | 0.6579 +/- 0.0859 | 0.0771 +/- 0.0252 | -0.7749 +/- 0.0873 | 0.6199 +/- 0.0699 |
| 5 stages | Base RMT n_mem=4 | 0.8910 +/- 0.0526 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1363 +/- 0.0657 | 0.1090 +/- 0.0526 |
| 5 stages | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.9448 +/- 0.0297 | 0.9635 +/- 0.0259 | 0.8175 +/- 0.1294 | -0.0234 +/- 0.0077 | 0.0187 +/- 0.0062 |
| 10 stages | Transformer/MHA | 0.0123 +/- 0.0071 | 0.5575 +/- 0.0162 | 0.0000 +/- 0.0000 | -0.6057 +/- 0.0107 | 0.5457 +/- 0.0099 |
| 10 stages | Base RMT n_mem=4 | 0.9032 +/- 0.0270 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1075 +/- 0.0300 | 0.0968 +/- 0.0270 |
| 10 stages | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.8747 +/- 0.0177 | 0.9000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0281 +/- 0.0197 | 0.0253 +/- 0.0177 |
| 20 stages | Transformer/MHA | 0.0081 +/- 0.0014 | 0.5371 +/- 0.1265 | 0.0000 +/- 0.0000 | -0.5569 +/- 0.1319 | 0.5291 +/- 0.1253 |
| 20 stages | Base RMT n_mem=4 | 0.7893 +/- 0.0420 | 0.9500 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.1692 +/- 0.0442 | 0.1607 +/- 0.0420 |
| 20 stages | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.8783 +/- 0.0264 | 0.9000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0228 +/- 0.0278 | 0.0217 +/- 0.0264 |

Interpretation:

- At 5 stages, this lower-budget run already shows the stability/plasticity
  tension. FMRMT retains best and has much less negative BWT than Base RMT, but
  its final-stage learning accuracy is below Base RMT.
- At 10 stages, Base RMT remains fully plastic through the final stage, while
  stable FMRMT learns stages 0-8 and then fails on stage 9. FMRMT's better BWT
  and forgetting numbers therefore reflect stronger retention of learned old
  stages, not better final-stage plasticity.
- At 20 stages, both recurrent models lose final-stage plasticity, but the
  failure pattern differs: Base RMT learns stages 0-18 and fails stage 19,
  whereas stable FMRMT learns stages 0-17 and fails stages 18-19. FMRMT still
  has much less old-stage degradation after learning.
- Transformer/MHA is mostly a weak baseline here. Its low seen accuracy combines
  old-stage forgetting with low learning accuracy on later stages.

## Formal Base RMT Memory-Token Control

Source: `results/class_incremental_ar_base_rmt_memory_sweep_formal_cl_20260517_040609.json`

Setup:

- Model family: Base RMT only.
- Swept `n_mem in {2, 4, 8, 16}`.
- Seeds: 123, 456, 789.
- Segment length: 64.
- Learning rate: 3e-3.
- `evaluate_future_stages=True`.
- 5-stage task uses the same seed list and budget as
  `results/class_incremental_ar_formal_cl_20260513_022211.json`.
- 10-stage task uses the same seed list and budget as the `10stage` condition in
  `results/class_incremental_ar_horizon_formal_cl_20260513_054644.json`.
- Stage-local random accuracy: 0.0625.

5-stage clean class-incremental AR:

| Model | Final Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=2 | 0.7599 +/- 0.0635 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.3002 +/- 0.0794 | 0.2401 +/- 0.0635 |
| Base RMT n_mem=4 | 0.8081 +/- 0.0147 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2399 +/- 0.0183 | 0.1919 +/- 0.0147 |
| Base RMT n_mem=8 | 0.6979 +/- 0.0604 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.3776 +/- 0.0755 | 0.3021 +/- 0.0604 |
| Base RMT n_mem=16 | 0.7973 +/- 0.0071 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2534 +/- 0.0089 | 0.2027 +/- 0.0071 |

10-stage horizon class-incremental AR:

| Model | Final Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=2 | 0.9042 +/- 0.0509 | 0.9979 +/- 0.0036 | 0.9792 +/- 0.0359 | -0.1041 +/- 0.0526 | 0.0937 +/- 0.0473 |
| Base RMT n_mem=4 | 0.9032 +/- 0.0270 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1075 +/- 0.0300 | 0.0968 +/- 0.0270 |
| Base RMT n_mem=8 | 0.8611 +/- 0.0490 | 0.9959 +/- 0.0070 | 0.9589 +/- 0.0699 | -0.1497 +/- 0.0467 | 0.1347 +/- 0.0421 |
| Base RMT n_mem=16 | 0.9314 +/- 0.0430 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0762 +/- 0.0478 | 0.0686 +/- 0.0430 |

Interpretation:

- `n_mem=4` remains the best Base RMT setting for the clean 5-stage formal task.
  It has the highest final seen average and least forgetting/BWT loss among the
  Base RMT memory sizes tested, with perfect learning accuracy.
- `n_mem=16` is the best Base RMT setting for the 10-stage horizon task. It
  keeps perfect final-stage plasticity and improves both seen average and BWT
  over the `n_mem=4` Base RMT baseline.
- `n_mem=8` is consistently not a good Base RMT setting in these controls. It is
  worst on 5-stage retention and also loses some late-stage plasticity on the
  10-stage horizon.
- The standard `Base RMT n_mem=4` vs stable `FMRMT n_mem=8` comparison is
  justified for the 5-stage formal task by this control. For the 10-stage
  horizon, `Base RMT n_mem=16` is the stronger Base RMT comparator and should be
  used in future plasticity/retention claims.
- Compared with stable FMRMT on the existing 10-stage horizon run, Base RMT
  `n_mem=16` has much better plasticity and higher final seen average, while
  stable FMRMT still has less negative BWT and lower forgetting from learning.
  That sharpens the retention/plasticity tradeoff rather than removing it.

## Formal Class-Incremental AR Interference Sweep

Source: `results/class_incremental_ar_interference_formal_cl_20260513_055618.json`

Setup:

- Task: 5-stage continual associative retrieval with stage-local test splits.
- Interference points:
  - `latest_updates0`: no repeated-key conflicts, latest-value target.
  - `latest_updates4`: 4 repeated-key conflicts, latest-value target.
  - `old_updates4`: 4 repeated-key conflicts, old-value target.
- New associations per stage: 16.
- Queries per sequence: 8.
- Sequence length: 128.
- Train examples per stage: 2048.
- Test examples per stage: 512.
- Epochs per stage: 8.
- Seeds: 123, 456, 789.
- Segment length for RMT variants: 64.
- `evaluate_future_stages=True`.
- Stage-local random accuracy: 0.0625.

| Condition | Model | Final Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No conflict, latest target | Transformer/MHA | 0.0380 +/- 0.0166 | 0.6579 +/- 0.0859 | 0.0771 +/- 0.0252 | -0.7749 +/- 0.0873 | 0.6199 +/- 0.0699 |
| No conflict, latest target | Base RMT n_mem=4 | 0.8910 +/- 0.0526 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1363 +/- 0.0657 | 0.1090 +/- 0.0526 |
| No conflict, latest target | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.9448 +/- 0.0297 | 0.9635 +/- 0.0259 | 0.8175 +/- 0.1294 | -0.0234 +/- 0.0077 | 0.0187 +/- 0.0062 |
| Repeated key, latest target | Transformer/MHA | 0.0232 +/- 0.0116 | 0.1985 +/- 0.0959 | 0.0583 +/- 0.0015 | -0.2191 +/- 0.1071 | 0.1753 +/- 0.0857 |
| Repeated key, latest target | Base RMT n_mem=4 | 0.1979 +/- 0.1776 | 0.6233 +/- 0.1273 | 0.4027 +/- 0.0693 | -0.5317 +/- 0.1269 | 0.4254 +/- 0.1015 |
| Repeated key, latest target | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.2388 +/- 0.1000 | 0.3784 +/- 0.0332 | 0.0971 +/- 0.0769 | -0.1744 +/- 0.0861 | 0.1551 +/- 0.0547 |
| Repeated key, old target | Transformer/MHA | 0.0554 +/- 0.0292 | 0.6299 +/- 0.0188 | 0.0735 +/- 0.0232 | -0.7181 +/- 0.0555 | 0.5745 +/- 0.0444 |
| Repeated key, old target | Base RMT n_mem=4 | 0.9023 +/- 0.0751 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1221 +/- 0.0938 | 0.0977 +/- 0.0751 |
| Repeated key, old target | FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.9397 +/- 0.0416 | 0.9609 +/- 0.0497 | 0.8043 +/- 0.2487 | -0.0264 +/- 0.0358 | 0.0211 +/- 0.0286 |

Interpretation:

- No-conflict latest-value results match the 5-stage horizon setting: FMRMT is
  the stronger retention model, while Base RMT is more consistently plastic in
  this lower-budget run.
- Repeated-key latest-value is the hard overwrite case. Base RMT has higher
  current-stage/learning accuracy, so it is more plastic here, but it pays for
  that with much more negative BWT and more forgetting after learning.
- The one-stage calibration above shows that repeated-key latest-value is not
  solved by any current model under this budget. Treat its low learning accuracy
  as within-stage overwrite difficulty plus possible continual plasticity
  pressure, not as clean continual plasticity loss by itself.
- Stable FMRMT has slightly higher final seen average on repeated-key latest
  value because it preserves learned stages better, but its low final plasticity
  is a low-current-stage-learning failure, not forgetting.
- Repeated-key old-value is easier for preservation. Base RMT learns perfectly
  but forgets more; FMRMT again has substantially better BWT/forgetting while
  showing weaker late-stage plasticity.


## Calibrated Fixed-Update Interference Task

Diagnostic screen sources:
`results/interference_ar_calibrated_one_stage_latest_mixed4_low_one_stage_latest_all4_low_one_stage_latest_all8_low_formal_cl_20260517_081359.json`
`results/interference_ar_calibrated_one_stage_latest_fixed_all8_low_formal_cl_20260517_081843.json`

Continual source:
`results/interference_ar_calibrated_continual_latest_fixed_all8_low_formal_cl_20260517_082621.json`

Setup:

- Task family: repeated-key latest-value associative retrieval with stage-local
  test splits.
- New calibrated condition: `update_value_mode="fixed_shift"`, where each
  repeated key's changed value is a deterministic within-stage value shift.
- All-updated condition: `num_query_associations=8` and
  `num_interference_pairs=8`, so every queried key has a misleading old value
  and a later changed value.
- Low budget: 2048 train examples/stage, 512 test examples/stage, 8 epochs/stage.
- Continual stages: 5.
- Seeds for continual run: 123, 456, 789.
- Models: Transformer/MHA, Base RMT `n_mem=16`, stable FMRMT
  `lr=0.005, slow_freq=4`, and plastic-tuned FMRMT
  `lr=0.01, slow_freq=1`.
- Stage-local random accuracy: 0.0625.

One-stage diagnostic accuracy, seed 123:

| One-Stage Condition | Transformer/MHA | Base RMT n_mem=16 | FMRMT Stable | FMRMT Plastic-Tuned |
| --- | ---: | ---: | ---: | ---: |
| Random updates, 4 of 8 queried keys updated | 0.4448 | 0.4963 | 0.5000 | 0.5000 |
| Random updates, all 8 queried keys updated | 0.0632 | 0.0681 | 0.0637 | 0.0762 |
| Fixed-shift updates, all 8 queried keys updated | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

5-stage fixed-shift all-updated latest-value continual run:

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0631 +/- 0.0446 | 0.6345 +/- 0.2215 | 0.1383 +/- 0.1106 | -0.7143 +/- 0.2341 | 0.5715 +/- 0.1873 | 14.8 +/- 0.1 |
| Base RMT n_mem=16 | 0.7878 +/- 0.0122 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2652 +/- 0.0152 | 0.2122 +/- 0.0122 | 24.7 +/- 0.3 |
| FMRMT stable lr=0.005 slow_freq=4 | 0.9553 +/- 0.0204 | 0.9597 +/- 0.0221 | 0.7987 +/- 0.1107 | -0.0056 +/- 0.0079 | 0.0045 +/- 0.0063 | 24.5 +/- 0.1 |
| FMRMT plastic lr=0.01 slow_freq=1 | 0.8549 +/- 0.0830 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1813 +/- 0.1038 | 0.1451 +/- 0.0830 | 25.4 +/- 0.2 |

Interpretation:

- The original random-update latest-value task was not a clean CL benchmark.
  With 4 of 8 queried keys updated, scores near 0.5 can come from retrieving
  unchanged old values and ignoring updates. When all queried keys are randomly
  updated, all models are near random, so the task becomes an unsolved in-context
  arbitrary-value copying problem.
- The fixed-shift all-updated variant is a better interference benchmark for the
  current model scale: every queried key has a misleading old value, the correct
  answer is the latest changed value, and all comparison models solve the
  one-stage task perfectly under the low budget.
- On the 5-stage fixed-shift continual task, Base RMT `n_mem=16` and
  plastic-tuned FMRMT are fully plastic: their final-stage and average learning
  accuracies are 1.0. Their lower seen averages come from old-stage degradation,
  i.e. forgetting / negative BWT.
- Stable FMRMT has the strongest retention by a large margin: BWT is near zero
  and forgetting from learning is near zero. Its weakness is lower current-stage
  learning on later stages, so this is a plasticity shortfall, not forgetting.
- This task is now a useful controlled overwrite benchmark: it preserves the
  retention/plasticity tradeoff while passing the one-stage solvability check.

### Budget16 Comet Rerun

Source: `results/interference_ar_calibrated_continual_latest_fixed_all8_budget16_comet_reconstructed_20260520_145536.json`

Original terminal log: `logs/calibrated_fixed_all8_budget16_comet_20260520_140157.log`

Caveat: this was launched through Comet, not the local JSON runner, so no raw
aggregate JSON was emitted by the experiment itself. The table below is
reconstructed from the final stage-local validation lines in the terminal log.
Comet reported metric throttling for several runs, so treat Comet batch-level
metric histories as potentially incomplete; the final validation stdout lines
used here are complete.

Setup difference from the low-budget run above: 4096 train examples/stage,
1024 test examples/stage, 16 epochs/stage, same 5-stage fixed-shift all-updated
latest-value task and seeds 123, 456, 789.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.1752 +/- 0.0594 | 0.9511 +/- 0.0387 | 0.7553 +/- 0.1937 | -0.9698 +/- 0.0261 | 0.7759 +/- 0.0209 |
| Base RMT n_mem=16 | 0.7972 +/- 0.0085 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2534 +/- 0.0106 | 0.2028 +/- 0.0085 |
| FMRMT stable lr=0.005 slow_freq=4 | 0.8679 +/- 0.0793 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1652 +/- 0.0991 | 0.1321 +/- 0.0793 |
| FMRMT plastic lr=0.01 slow_freq=1 | 0.7474 +/- 0.0460 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.3158 +/- 0.0575 | 0.2526 +/- 0.0460 |

Interpretation:

- With the larger per-stage budget, all RMT variants are fully plastic on this
  calibrated interference task. Stable FMRMT's low-budget late-stage plasticity
  issue disappears here, so the remaining difference is retention.
- Stable FMRMT keeps the best seen average and least forgetting, but the margin
  over Base RMT is smaller than in the low-budget run and no longer near-zero
  forgetting.
- Plastic-tuned FMRMT is the worst RMT variant here: it preserves full
  plasticity but forgets more than Base RMT and stable FMRMT.
- Transformer/MHA can learn many current stages under this larger budget, but
  final seen accuracy remains poor because old stages collapse almost completely.
- For this benchmark, the next useful question is not whether FMRMT can learn
  the current stage at sufficient budget; it can. The useful question is whether
  a middle setting can keep stable-FMRMT retention while avoiding low-budget
  plasticity failures.


## Selected Formal Timing Reruns

Source: `results/class_incremental_ar_timing_selected_formal_cl_20260517_055216.json`

Setup:

- Timing instrumentation: `train_continual()` stage-level wall time, processed
  examples/tokens, batches, optimizer steps, and throughput metrics.
- Hardware: GPU 0, `NVIDIA GeForce GTX 1080 Ti`.
- Seeds: 123, 456, 789.
- Segment length: 64 for RMT variants.
- Stage-local random accuracy: 0.0625.
- 5-stage clean rerun matches the formal 5-stage budget:
  4096 train examples/stage, 1024 test examples/stage, 16 epochs/stage.
- 10-stage horizon rerun matches the 10-stage horizon budget:
  2048 train examples/stage, 512 test examples/stage, 8 epochs/stage.
- Wall-clock times are serial-run measurements on this machine; use them as
  local relative costs, not architecture-intrinsic FLOP estimates.

5-stage clean class-incremental AR:

| Model | Seen Avg Accuracy | Plasticity | Avg BWT | Total Wall Seconds | Train Seconds | Seen-Eval Seconds | Train Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.1321 +/- 0.1024 | 0.4832 +/- 0.3324 | -0.9504 +/- 0.0455 | 57.8 +/- 0.2 | 55.3 +/- 0.2 | 2.4 +/- 0.0 | 11847 +/- 39 |
| Base RMT n_mem=4 | 0.8081 +/- 0.0147 | 1.0000 +/- 0.0000 | -0.2399 +/- 0.0183 | 93.8 +/- 0.3 | 91.0 +/- 0.3 | 2.8 +/- 0.0 | 7206 +/- 23 |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.8501 +/- 0.0488 | 1.0000 +/- 0.0000 | -0.1874 +/- 0.0609 | 95.4 +/- 0.1 | 92.6 +/- 0.1 | 2.8 +/- 0.0 | 7081 +/- 7 |

10-stage horizon class-incremental AR:

| Model | Seen Avg Accuracy | Plasticity | Avg BWT | Total Wall Seconds | Train Seconds | Seen-Eval Seconds | Train Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=4 | 0.9032 +/- 0.0270 | 1.0000 +/- 0.0000 | -0.1075 +/- 0.0300 | 51.0 +/- 0.2 | 45.7 +/- 0.2 | 5.3 +/- 0.0 | 7172 +/- 32 |
| Base RMT n_mem=16 | 0.9314 +/- 0.0430 | 1.0000 +/- 0.0000 | -0.0762 +/- 0.0478 | 51.7 +/- 0.0 | 46.3 +/- 0.0 | 5.4 +/- 0.0 | 7085 +/- 6 |
| FMRMT n_mem=8, lr=0.005, slow_freq=4 | 0.8747 +/- 0.0177 | 0.0000 +/- 0.0000 | -0.0281 +/- 0.0197 | 51.4 +/- 0.2 | 46.2 +/- 0.2 | 5.3 +/- 0.0 | 7102 +/- 29 |

Interpretation:

- On the 5-stage clean task, FMRMT's retention gain over Base RMT is not free,
  but the local cost is small: about 95.4s vs 93.8s total wall time, or roughly
  1.7% slower on this GPU. Both are much slower than Transformer/MHA, but
  Transformer/MHA is not a useful accuracy baseline here.
- On the 10-stage horizon task, Base RMT `n_mem=16` is the strongest
  compute-aware Base RMT comparator: it improves seen accuracy and BWT over
  `n_mem=4`, keeps full plasticity, and costs only about 0.7s more total wall
  time in this run.
- Stable FMRMT on 10-stage has essentially the same local wall time as Base RMT
  `n_mem=16`, but it fails final-stage plasticity. Its better BWT therefore
  reflects retention of learned stages, not a compute-efficient win.
- For future claims, 5-stage comparisons can keep Base RMT `n_mem=4`; 10-stage
  horizon comparisons should use Base RMT `n_mem=16` as the main Base RMT
  baseline and report timing alongside accuracy/BWT.


## FMRMT Plasticity Tuning on 10-Stage Horizon

Screen source:
`results/fmrmt_plasticity_tuning_horizon10_formal_cl_20260517_063958.json`

Confirmation source:
`results/fmrmt_plasticity_confirm_horizon10_slow1_formal_cl_20260517_065953.json`

Setup:

- Task: 10-stage class-incremental associative retrieval with stage-local test
  splits, matching the 10-stage horizon condition above.
- FMRMT: `n_mem=8`, `reset_memory_each_batch=False`,
  `rmt_clip_memory_grad=1.0`, segment length 64.
- Screen: seed 123, `rmt_fast_lr in {0.001, 0.005, 0.01, 0.02}`,
  `rmt_slow_update_freq in {1, 2, 4}`,
  `reset_memory_each_epoch in {True, False}`.
- Confirmation: seeds 123, 456, 789, `rmt_slow_update_freq=1`,
  `rmt_fast_lr in {0.001, 0.005, 0.01}`,
  `reset_memory_each_epoch in {True, False}`.
- Hardware: GPU 0, `NVIDIA GeForce GTX 1080 Ti`.
- Stage-local random accuracy: 0.0625.

Seed-123 screen, best plasticity-first config per slow update frequency:

| Slow Freq | Fast LR | Epoch Reset | Seen Avg Accuracy | Avg Learning Accuracy | Plasticity | Avg BWT | Forgetting | Total Wall Seconds |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.001 | True | 0.9267 | 1.0000 | 1.0000 | -0.0814 | 0.0733 | 53.9 |
| 2 | 0.020 | True | 0.9176 | 0.9802 | 0.8015 | -0.0695 | 0.0625 | 53.1 |
| 4 | 0.001 | False | 0.8724 | 0.9000 | 0.0000 | -0.0307 | 0.0276 | 52.3 |

Three-seed confirmation for `rmt_slow_update_freq=1`:

| Fast LR | Epoch Reset | Seen Avg Accuracy | Avg Learning Accuracy | Plasticity | Avg BWT | Forgetting | Total Wall Seconds | Train Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.010 | True | 0.8977 +/- 0.0244 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1137 +/- 0.0271 | 0.1023 +/- 0.0244 | 54.0 +/- 0.0 | 6743 +/- 16 |
| 0.001 | True | 0.8852 +/- 0.0363 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1276 +/- 0.0403 | 0.1148 +/- 0.0363 | 53.8 +/- 0.2 | 6760 +/- 28 |
| 0.001 | False | 0.8590 +/- 0.0703 | 0.9989 +/- 0.0018 | 0.9994 +/- 0.0008 | -0.1554 +/- 0.0786 | 0.1399 +/- 0.0708 | 53.6 +/- 0.2 | 6787 +/- 33 |
| 0.005 | False | 0.9091 +/- 0.0492 | 0.9974 +/- 0.0045 | 0.9743 +/- 0.0445 | -0.0981 +/- 0.0577 | 0.0883 +/- 0.0520 | 53.5 +/- 0.1 | 6799 +/- 14 |
| 0.005 | True | 0.9055 +/- 0.0423 | 0.9965 +/- 0.0031 | 0.9653 +/- 0.0308 | -0.1012 +/- 0.0464 | 0.0911 +/- 0.0417 | 53.9 +/- 0.1 | 6749 +/- 19 |
| 0.010 | False | 0.8055 +/- 0.1507 | 0.9796 +/- 0.0352 | 0.7957 +/- 0.3523 | -0.1934 +/- 0.1293 | 0.1741 +/- 0.1164 | 53.6 +/- 0.1 | 6783 +/- 17 |

Interpretation:

- `rmt_slow_update_freq=1` is the main knob that restores late-stage
  plasticity. In the seed-123 screen, all `slow_freq=4` configs had final
  plasticity 0.0, matching the stable-FMRMT late-stage failure; `slow_freq=2`
  partially helped but did not solve the final stage.
- The best confirmed plasticity-first FMRMT setting on horizon10 is
  `rmt_fast_lr=0.01`, `rmt_slow_update_freq=1`,
  `reset_memory_each_epoch=True`. It reaches perfect final plasticity and
  perfect average learning accuracy across 3 seeds.
- This tuning improves over stable FMRMT on final-stage learning
  (`1.0000` vs `0.0000` plasticity) and raises final seen average
  (`0.8977` vs `0.8747`), but it worsens old-stage retention
  (`-0.1137` vs `-0.0281` BWT). That is a plasticity/retention tradeoff, not a
  pure improvement.
- Against the stronger 10-stage Base RMT comparator (`n_mem=16`), tuned FMRMT is
  not a Pareto win: Base RMT has higher seen accuracy (`0.9314`), perfect
  plasticity, better BWT (`-0.0762`), and lower wall time (`51.7s` vs `54.0s`).
- The `lr=0.005` settings are retention-competitive, but they miss the final
  stage on some seeds. The no-epoch-reset settings are more variable, especially
  `lr=0.01`, which failed badly on seed 789.

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
