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

## Formal 20-Stage Permuted AR Scheduler Control

Sources:
`results/class_incremental_ar_permuted_formal_global_cosine_formal20_permuted_20260523_231700.json`
`results/class_incremental_ar_permuted_formal_stage_cosine_formal20_permuted_20260523_231700.json`
`results/class_incremental_ar_permuted_formal_constant_formal20_permuted_20260523_231649.json`

Setup:

- Task: 20-stage class-incremental AR with fixed per-stage random value
  permutations, `association_table_seed=20260522`.
- Seeds: 123, 456, 789.
- Train examples/stage: 2048.
- Test examples/stage: 512.
- Epochs/stage: 8.
- Sequence length: 128.
- Segment length for RMT variants: 64.
- Per-epoch current-stage validation enabled.
- Scheduler modes:
  - `global_cosine`: one cosine schedule across all continual stages.
  - `stage_cosine`: LR reset to base value at each stage, then cosine over the
    stage's local epochs.
  - `constant`: fixed LR throughout training.

| Scheduler | Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| global cosine | Transformer/MHA | 0.0153 +/- 0.0016 | 0.5663 +/- 0.0264 | 0.0000 +/- 0.0000 | -0.5800 +/- 0.0277 | 0.5510 +/- 0.0263 |
| global cosine | Base RMT n_mem=16 | 0.8497 +/- 0.0144 | 0.9500 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.1055 +/- 0.0151 | 0.1003 +/- 0.0144 |
| global cosine | FMRMT stable lr=0.005 slow_freq=4 | 0.8945 +/- 0.0035 | 0.9000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0058 +/- 0.0037 | 0.0055 +/- 0.0035 |
| global cosine | FMRMT plastic lr=0.01 slow_freq=1 | 0.8201 +/- 0.0141 | 0.9500 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.1367 +/- 0.0148 | 0.1299 +/- 0.0141 |
| stage cosine | Transformer/MHA | 0.0100 +/- 0.0040 | 0.4536 +/- 0.1096 | 0.1829 +/- 0.0981 | -0.4670 +/- 0.1169 | 0.4436 +/- 0.1111 |
| stage cosine | Base RMT n_mem=16 | 0.0647 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.9845 +/- 0.0077 | 0.9353 +/- 0.0073 |
| stage cosine | FMRMT stable lr=0.005 slow_freq=4 | 0.2815 +/- 0.0824 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.7563 +/- 0.0867 | 0.7185 +/- 0.0824 |
| stage cosine | FMRMT plastic lr=0.01 slow_freq=1 | 0.1179 +/- 0.0266 | 0.9941 +/- 0.0084 | 1.0000 +/- 0.0000 | -0.9223 +/- 0.0193 | 0.8762 +/- 0.0184 |
| constant | Transformer/MHA | 0.0258 +/- 0.0277 | 0.6626 +/- 0.0880 | 0.3757 +/- 0.4413 | -0.6704 +/- 0.0738 | 0.6369 +/- 0.0701 |
| constant | Base RMT n_mem=16 | 0.2145 +/- 0.1111 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.8268 +/- 0.1169 | 0.7855 +/- 0.1111 |
| constant | FMRMT stable lr=0.005 slow_freq=4 | 0.2165 +/- 0.1180 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.8247 +/- 0.1242 | 0.7835 +/- 0.1180 |
| constant | FMRMT plastic lr=0.01 slow_freq=1 | 0.1498 +/- 0.0584 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.8950 +/- 0.0614 | 0.8502 +/- 0.0584 |

Interpretation:

- The abrupt final-stage plasticity collapse in the 20-stage permuted run is a
  scheduler confound under the current setup. With `global_cosine`, all RMT
  variants have final plasticity 0.0; with `stage_cosine` or `constant`, the
  RMT variants recover final-stage learning.
- This means the late zero-plasticity result should not be used as evidence of
  intrinsic plasticity loss. It mostly reflects the global LR schedule reaching
  a near-zero learning rate at the end of the continual stream.
- The scheduler control exposes a sharper stability/plasticity tradeoff:
  resetting or holding LR restores current-stage learning, but old-stage
  retention collapses. Under `stage_cosine`, Base RMT and both FMRMT variants
  learn the final stage, but BWT is very negative and final seen accuracy is low.
- `global_cosine` is therefore a strong stability/retention setting but a bad
  plasticity diagnostic at long horizons. Future paper claims should separate
  LR-schedule-induced late learning failure from genuine model plasticity loss.
- The new JSON artifacts include full metric history, stage-end matrices, and
  per-epoch current-stage learning curves for heatmap and learning-curve plots.

## Formal 20-Stage Permuted AR Slow-Update Accumulation Control

Sources:
`results/class_incremental_ar_permuted_formal_global_cosine_accumulate_formal20_permuted_20260524_023612.json`
`results/class_incremental_ar_permuted_formal_global_cosine_accumulate_formal20_permuted_20260524_023618.json`
`results/class_incremental_ar_permuted_formal_global_cosine_accumulate_formal20_permuted_20260524_030056_759787.json`

Setup: same 20-stage permuted AR task as above, `global_cosine` scheduler,
`slow_update_mode=accumulate`, seeds 123/456/789. The result writer now uses
microsecond timestamps so parallel per-seed runs do not collide.

| Model | Slow Optimizer Steps/Stage | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds | Stage 18 Plasticity | Stage 19 Plasticity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=16 | 256 | 0.8497 +/- 0.0144 | 0.9500 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.1055 +/- 0.0151 | 0.1003 +/- 0.0144 | 122.4 +/- 3.6 | 1.0000 | 0.0000 |
| FMRMT stable lr=0.005 slow_freq=4 | 64 | 0.8843 +/- 0.0070 | 0.9000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0165 +/- 0.0074 | 0.0157 +/- 0.0070 | 122.0 +/- 5.1 | 0.0000 | 0.0000 |
| FMRMT no-fast lr=0 slow_freq=4 | 64 | 0.8927 +/- 0.0025 | 0.9000 +/- 0.0000 | 0.0000 +/- 0.0000 | -0.0077 +/- 0.0026 | 0.0073 +/- 0.0025 | 122.0 +/- 5.3 | 0.0000 | 0.0000 |
| FMRMT lr=0.005 slow_freq=2 | 128 | 0.8870 +/- 0.0183 | 0.9454 +/- 0.0031 | 0.0000 +/- 0.0000 | -0.0615 +/- 0.0196 | 0.0589 +/- 0.0186 | 122.8 +/- 4.9 | 0.9082 | 0.0000 |

Interpretation:

- Gradient accumulation makes the AR slow-update semantics match the CIFAR-style
  two-timescale implementation more closely, but it does not change the
  20-stage `global_cosine` conclusion: every model still has zero stage-19
  plasticity because the global LR reaches 0 at the final stage.
- Accumulation is not a hidden fix for stable FMRMT. Compared with the prior
  `skip` run, stable FMRMT is slightly worse on retention
  (`seen 0.8843` vs `0.8945`, BWT `-0.0165` vs `-0.0058`) and still fails
  stages 18-19.
- `rmt_slow_update_freq=2` is the useful middle setting: it learns stage 18
  on average (`0.9082`) and keeps high average learning accuracy (`0.9454`),
  but it gives up retention relative to slow_freq=4. This is still the same
  retention/plasticity tradeoff, not a clean win.
- The `rmt_fast_lr=0` slow_freq=4 control is close to stable FMRMT and slightly
  better on these aggregate retention metrics. Under this specific global LR
  setup, the retention advantage is therefore not evidence that the fast-memory
  gradient update itself is helping; it may be mostly the slow-update cadence
  and memory-augmented architecture.

## Formal 20-Stage Permuted AR Stage-OneCycle Control

Sources:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033410_720075.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033413_865324.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033421_414890.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_109006.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_124162.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_814041.json`

Setup: same 20-stage permuted AR task as above, `slow_update_mode=accumulate`,
seeds 123/456/789. The scheduler is a CIFAR-style stage-local linear
`OneCycleLR`: it resets at every continual stage, uses the actual number of
slow optimizer steps per epoch, and steps only when the slow optimizer steps.

| Model | Slow Optimizer Steps/Stage | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds | Stage 18 Plasticity | Stage 19 Plasticity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 256 | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 84.8 +/- 0.1 | 0.4795 | 0.7928 |
| Base RMT n_mem=16 | 256 | 0.8121 +/- 0.0120 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1978 +/- 0.0126 | 0.1879 +/- 0.0120 | 125.0 +/- 0.8 | 1.0000 | 1.0000 |
| FMRMT stable lr=0.005 slow_freq=4 | 64 | 0.9330 +/- 0.0431 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0705 +/- 0.0454 | 0.0670 +/- 0.0431 | 125.5 +/- 1.3 | 1.0000 | 1.0000 |
| FMRMT no-fast lr=0 slow_freq=4 | 64 | 0.9114 +/- 0.0492 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0933 +/- 0.0518 | 0.0886 +/- 0.0492 | 125.6 +/- 1.3 | 1.0000 | 1.0000 |
| FMRMT lr=0.005 slow_freq=2 | 128 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 | 126.1 +/- 1.0 | 1.0000 | 1.0000 |
| FMRMT no-fast lr=0 slow_freq=2 | 128 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 | 123.5 +/- 0.1 | 1.0000 | 1.0000 |

Interpretation:

- Stage-local OneCycle removes the global-cosine late-LR collapse without the
  severe forgetting seen under `stage_cosine` or `constant`. All tested RMT
  variants reach perfect final-stage plasticity and perfect average learning
  accuracy.
- The strongest settings are the slow_freq=2 FMRMT variants. Both fast-memory
  and no-fast versions reach nearly identical retention: seen accuracy
  `0.9788` vs `0.9774`, BWT `-0.0224` vs `-0.0238`. Under this recipe, the
  improvement over Base RMT is real, but it is not attributable to the fast
  memory gradient update alone.
- The matched slow_freq=4 control still weakly favors the fast-memory update
  (`seen 0.9330` vs `0.9114`, BWT `-0.0705` vs `-0.0933`), but the larger and
  cleaner effect is the slow-update cadence combined with the memory-augmented
  architecture.
- Transformer/MHA benefits from the stage-local schedule for current-stage
  learning, but it remains a poor continual baseline: final seen accuracy is
  only `0.0419` and BWT is `-0.8588`, meaning old stages mostly collapse.
- The defensible current paper claim is schedule-aware: on 20-stage permuted AR,
  stage-local OneCycle plus RMT-style memory and slow updates gives both high
  plasticity and strong retention; the local fast-memory update is not yet the
  isolated cause of the best result.

### Stage-OneCycle Fast-LR Sweep at Slow Freq 2

Sources:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_044509_486077.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_044509_507492.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_044509_723219.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_050637_068951.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_050638_265093.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_050641_263478.json`

Setup: same 20-stage permuted AR task, `stage_onecycle`,
`slow_update_mode=accumulate`, `rmt_slow_update_freq=2`, FMRMT `n_mem=8`,
`reset_memory_each_epoch=True`, `reset_memory_each_batch=False`, seeds
123/456/789. The first sweep covered `rmt_fast_lr <= 0.02`; the second was a
high-LR stress check at `0.05`, `0.1`, and `0.2`.

| Fast LR | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds | Stage 18 Plasticity | Stage 19 Plasticity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 | 123.5 +/- 0.1 | 1.0000 | 1.0000 |
| 0.001 | 0.9764 +/- 0.0135 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0248 +/- 0.0142 | 0.0236 +/- 0.0135 | 123.9 +/- 0.1 | 1.0000 | 1.0000 |
| 0.002 | 0.9764 +/- 0.0150 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0248 +/- 0.0158 | 0.0236 +/- 0.0150 | 124.7 +/- 0.0 | 1.0000 | 1.0000 |
| 0.005 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 | 125.0 +/- 0.1 | 1.0000 | 1.0000 |
| 0.010 | 0.9701 +/- 0.0043 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0314 +/- 0.0045 | 0.0299 +/- 0.0043 | 124.3 +/- 0.1 | 1.0000 | 1.0000 |
| 0.020 | 0.9756 +/- 0.0032 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0257 +/- 0.0033 | 0.0244 +/- 0.0032 | 124.5 +/- 0.1 | 1.0000 | 1.0000 |
| 0.050 | 0.9643 +/- 0.0108 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0376 +/- 0.0114 | 0.0357 +/- 0.0108 | 125.6 +/- 0.6 | 1.0000 | 1.0000 |
| 0.100 | 0.9402 +/- 0.0356 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0629 +/- 0.0375 | 0.0598 +/- 0.0356 | 125.9 +/- 0.7 | 1.0000 | 1.0000 |
| 0.200 | 0.9350 +/- 0.0238 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0684 +/- 0.0251 | 0.0650 +/- 0.0238 | 125.9 +/- 0.6 | 1.0000 | 1.0000 |

Interpretation:

- All settings in this slow_freq=2, stage-local-OneCycle regime solve
  current-stage learning: average learning accuracy, final plasticity, and
  stage-18/stage-19 plasticity are all 1.0.
- The best nonzero fast LR by mean seen accuracy and BWT is `0.005`, matching
  the prior default. However, the margin over the no-fast control is tiny
  (`seen 0.9788` vs `0.9774`, BWT `-0.0224` vs `-0.0238`) and well within
  seed variability.
- There is no monotonic fast-LR effect. `0.01` is the weakest point in this
  sweep, while `0.02` is stable but not better than `0.005`.
- Very large fast LRs do not break plasticity, likely because the memory
  gradient is clipped, but they do degrade retention. `0.05`, `0.1`, and `0.2`
  all have worse seen accuracy and BWT than `0.005` and the no-fast control.
- This sweep supports using `rmt_fast_lr=0.005` as the best nonzero option for
  this recipe, but it does not support a strong claim that the local fast-memory
  update is necessary for the 20-stage permuted AR result.

### Segment-Length 128 Control

Sources:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_segment_len128_formal20_permuted_20260524_223513_274415.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_segment_len128_formal20_permuted_20260524_223514_442283.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_segment_len128_formal20_permuted_20260524_223517_318588.json`

Setup: formal20 permuted AR, stage-local `OneCycleLR`,
`slow_update_mode=accumulate`, seeds 123/456/789. The usual setting uses
`input_seq_len=128` and `segment_len=64`, so RMT variants process each sample as
two recurrent segments. This control sets `segment_len=128`, so each sample is
processed as one segment. It removes cross-segment recurrence while preserving
the memory-token architecture inside the RMT block.

| Model | Segment Len | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 64 | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 84.8 +/- 0.1 |
| Transformer/MHA | 128 | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 85.4 +/- 0.3 |
| Base RMT n_mem=16 | 64 | 0.8121 +/- 0.0120 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1978 +/- 0.0126 | 0.1879 +/- 0.0120 | 125.0 +/- 0.8 |
| Base RMT n_mem=16 | 128 | 0.7920 +/- 0.0686 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2189 +/- 0.0722 | 0.2080 +/- 0.0686 | 111.7 +/- 0.3 |
| FMRMT lr=0.005 slow_freq=2 | 64 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 | 125.0 +/- 0.1 |
| FMRMT lr=0.005 slow_freq=2 | 128 | 0.9144 +/- 0.0166 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0901 +/- 0.0175 | 0.0856 +/- 0.0166 | 111.1 +/- 0.2 |
| FMRMT fast_lr=0 slow_freq=2 | 64 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 | 123.5 +/- 0.1 |
| FMRMT fast_lr=0 slow_freq=2 | 128 | 0.9432 +/- 0.0207 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0598 +/- 0.0217 | 0.0568 +/- 0.0207 | 110.9 +/- 0.2 |

Interpretation:

- Cross-segment recurrence is not required for the broad RMT-vs-Transformer
  separation. Even with `segment_len=128`, Base RMT reaches `0.7920` seen
  accuracy and both FMRMT slow2 variants remain far above Transformer/MHA.
- Cross-segment recurrence does improve retention for the strongest FMRMT
  settings. Moving from `segment_len=64` to `128` drops FMRMT lr=0.005 from
  `0.9788` to `0.9144` seen accuracy and worsens BWT from `-0.0224` to
  `-0.0901`.
- The no-fast control remains competitive and is actually better than nonzero
  fast LR at `segment_len=128` (`0.9432` vs `0.9144` seen accuracy). This
  further weakens any AR-only claim that the fast-memory gradient update is the
  isolated source of the gain.
- The careful mechanism claim for formal20 permuted AR is therefore:
  memory-augmented RMT blocks plus the current two-timescale/slow-update recipe
  give strong retention; two-segment recurrence helps the best FMRMT setting,
  but the local AR evidence still does not isolate fast-memory updates as the
  causal factor.

### Longer Sequence / More Associations Stress Diagnostic

Source:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq256_seg64_vocab2048_assoc32_query16_train1024_formal20_permuted_20260525_010326_762385.json`

Setup: exploratory one-seed run (`seed=123`) on formal20 permuted AR with
`input_seq_len=256`, `segment_len=64`, `vocab_size=2048`,
`associations_per_stage=32`, `num_query_associations=16`, 1024 train examples
per stage, 512 test examples per stage, 8 epochs per stage,
stage-local `OneCycleLR`, and `slow_update_mode=accumulate`. Random
stage-local accuracy is `1 / 32 = 0.03125`.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0016 | 0.1396 | 0.0311 | -0.1453 | 0.1381 | 98.6 |
| Base RMT n_mem=16 | 0.9958 | 1.0000 | 1.0000 | -0.0044 | 0.0042 | 133.8 |
| FastMem RMT lr=0.005 slow_freq=2 | 0.9909 | 1.0000 | 1.0000 | -0.0096 | 0.0091 | 128.9 |
| FastMem RMT fast_lr=0 slow_freq=2 | 0.9598 | 1.0000 | 1.0000 | -0.0423 | 0.0402 | 128.5 |

Interpretation:

- This setting is much harder for the regular Transformer: final plasticity is
  at random chance and cumulative seen accuracy collapses.
- It is still not hard enough to separate Base RMT from the FastMem variants:
  Base RMT solves the diagnostic almost perfectly.
- Unlike the default formal20 setting, the nonzero FastMem update does beat the
  no-fast control on seed 123 (`seen 0.9909` vs `0.9598`, BWT `-0.0096` vs
  `-0.0423`). This is a useful signal, but it is not yet a paper-level claim
  because it is one seed and Base RMT remains strongest.
- The next stress diagnostic should reduce slow-weight opportunity or increase
  association load further, rather than only increasing vocabulary. Good next
  candidates are `train_examples=512`, `input_seq_len=512` with
  `segment_len=64`, or `associations_per_stage=64`.

Follow-up source with lower train budget:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq256_seg64_vocab2048_assoc32_query16_train512_formal20_permuted_20260525_011431_447484.json`

Setup is identical except 512 train examples per stage and only the RMT-family
models were run (`seed=123`).

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=16 | 0.9982 | 1.0000 | 1.0000 | -0.0019 | 0.0018 | 92.5 |
| FastMem RMT lr=0.005 slow_freq=2 | 0.9112 | 1.0000 | 1.0000 | -0.0935 | 0.0888 | 88.8 |
| FastMem RMT fast_lr=0 slow_freq=2 | 0.9313 | 1.0000 | 1.0000 | -0.0723 | 0.0687 | 89.1 |

Interpretation:

- Reducing the train budget does not expose a fast-update advantage on this
  seed. It instead makes the nonzero FastMem update worse than the no-fast
  control (`seen 0.9112` vs `0.9313`).
- This is retention degradation, not plasticity loss: all three models have
  average learning accuracy and final plasticity equal to 1.0.
- The retention loss is concentrated in early stages. At the final checkpoint,
  FastMem RMT lr=0.005 has stage-0/stage-1 accuracies `0.2554/0.0490`, while
  the no-fast control has `0.5306/0.1797`. Base RMT keeps all stages near
  perfect.
- This suggests the current FastMem RMT setting may be over-updating or
  destabilizing learned early-stage memory in this stress regime. The next
  useful diagnostic is either a smaller fast LR (`0.001` or `0.002`) at the
  same stress setting or a stricter architecture-matched control that compares
  Base RMT with `n_mem=8` against FastMem/no-fast `n_mem=8`.

Matched-memory follow-up source:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq256_seg64_vocab2048_assoc32_query16_train512_nmem8_fastlr_formal20_permuted_20260525_013010_914398.json`

Setup is the same train512 stress diagnostic (`seed=123`) but compares Base RMT
with the same carried memory-state size as the FastMem variants (`n_mem=8`) and
sweeps smaller nonzero fast LRs.

| Model | Fast LR | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=8 | n/a | 0.9614 | 1.0000 | 1.0000 | -0.0406 | 0.0386 | 87.2 |
| FastMem RMT n_mem=8 | 0.000 | 0.9313 | 1.0000 | 1.0000 | -0.0723 | 0.0687 | 88.6 |
| FastMem RMT n_mem=8 | 0.001 | 0.9168 | 1.0000 | 1.0000 | -0.0875 | 0.0832 | 88.3 |
| FastMem RMT n_mem=8 | 0.002 | 0.9260 | 1.0000 | 1.0000 | -0.0779 | 0.0740 | 88.1 |
| FastMem RMT n_mem=8 | 0.005 | 0.9112 | 1.0000 | 1.0000 | -0.0935 | 0.0888 | 88.0 |

Interpretation:

- Matching carried memory-state size changes the Base RMT comparison
  substantially. Base RMT n_mem=8 drops from the prior n_mem=16 result
  (`seen 0.9982`) to `0.9614`, so part of the Base RMT advantage in the
  train512 stress diagnostic was memory capacity.
- Even after matching memory size, Base RMT n_mem=8 is still better than the
  FastMem-style variants on this seed.
- No nonzero fast LR helps here. The no-fast FastMem control is the best
  FastMem-style setting (`seen 0.9313`), while `0.001`, `0.002`, and `0.005`
  all have worse retention.
- Plasticity is again saturated for every RMT-family model. The differences are
  retention/forgetting differences, not current-stage learning failures.
- The main residual uncertainty is implementation-specific: FastMem RMT uses
  `eval_memory_policy="initial"` and `reset_memory_each_epoch=True`, so the
  final stage-local tests evaluate the learned initial memory rather than the
  carried fast state. If we want to test whether the fast state itself is useful
  as a continual memory, the next diagnostic should repeat this matched-memory
  stress sweep with `eval_memory_policy="fast"` and/or
  `reset_memory_each_epoch=False`.

Matched-memory eval/reset policy follow-up source:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq256_seg64_vocab2048_assoc32_query16_train512_evalfast_noepochreset_formal20_permuted_20260525_015226_938146.json`

Setup is the same train512 stress diagnostic (`seed=123`, `n_mem=8`) and tests
the two remaining FastMem RMT state-policy questions separately and together:
evaluate the carried fast memory directly with `eval_memory_policy="fast"`, let
fast memory persist across epochs with `reset_memory_each_epoch=False`, and use
both at once.

| Model | Eval Memory | Reset Each Epoch | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RMT n_mem=8 | n/a | n/a | 0.9614 | 1.0000 | 1.0000 | -0.0406 | 0.0386 | 86.0 |
| FastMem RMT lr=0.005 slow_freq=2 | initial | True | 0.9112 | 1.0000 | 1.0000 | -0.0935 | 0.0888 | 87.6 |
| FastMem RMT lr=0.005 slow_freq=2 | fast | True | 0.9112 | 1.0000 | 1.0000 | -0.0935 | 0.0888 | 87.3 |
| FastMem RMT lr=0.005 slow_freq=2 | initial | False | 0.9106 | 1.0000 | 1.0000 | -0.0941 | 0.0894 | 87.3 |
| FastMem RMT lr=0.005 slow_freq=2 | fast | False | 0.9094 | 1.0000 | 1.0000 | -0.0954 | 0.0906 | 87.9 |

Interpretation:

- Evaluating the carried fast state directly does not improve this setting. With
  epoch resets enabled, `eval_memory_policy="fast"` is indistinguishable from
  the default initial-memory evaluation at aggregate and stage-local accuracy.
- Letting fast memory persist across epochs and stages also does not help. It is
  slightly worse under initial-memory evaluation (`seen 0.9106`) and slightly
  worse again when the carried fast state is evaluated directly (`seen 0.9094`).
- The early-stage retention pattern remains the problem. Base RMT n_mem=8 ends
  with stage-0/stage-1 accuracies `0.4185/0.9440`; default FastMem RMT ends at
  `0.2554/0.0490`; no-epoch-reset FastMem RMT improves these to
  `0.3242/0.1887` but loses enough elsewhere that average seen accuracy does
  not improve.
- This closes the most direct state-policy explanation for the stress result:
  the current FastMem RMT update is not hidden by the default eval/reset policy.
  On this diagnostic, the nonzero fast update still hurts retention relative to
  Base RMT n_mem=8 and to the no-fast FastMem control.

Matched-memory seq512 stress sources:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq512_seg64_vocab4096_assoc32_query16_train512_formal20_permuted_20260525_022546_765099.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_stress_seq512_seg64_vocab4096_assoc64_query16_train512_formal20_permuted_20260525_022546_234926.json`

Setup: exploratory one-seed runs (`seed=123`) on formal20 permuted AR with
`input_seq_len=512`, `segment_len=64`, `vocab_size=4096`, 512 train examples
per stage, 512 test examples per stage, 8 epochs per stage, stage-local
`OneCycleLR`, and `slow_update_mode=accumulate`. Shape B uses
`associations_per_stage=32`, `num_query_associations=16`; shape C uses
`associations_per_stage=64`, `num_query_associations=16`.

| Shape | Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B: assoc32 | Transformer/MHA | 0.0015 | 0.1873 | 0.0295 | -0.1956 | 0.1858 | 162.9 |
| B: assoc32 | Base RMT n_mem=16 | 0.9449 | 1.0000 | 1.0000 | -0.0580 | 0.0551 | 187.4 |
| B: assoc32 | Base RMT n_mem=8 | 0.8893 | 1.0000 | 1.0000 | -0.1165 | 0.1107 | 177.9 |
| B: assoc32 | FastMem RMT fast_lr=0 slow_freq=2 | 0.8987 | 1.0000 | 1.0000 | -0.1066 | 0.1013 | 178.5 |
| B: assoc32 | FastMem RMT lr=0.005 slow_freq=2 | 0.9191 | 1.0000 | 1.0000 | -0.0851 | 0.0809 | 178.5 |
| C: assoc64 | Transformer/MHA | 0.0008 | 0.1144 | 0.0164 | -0.1196 | 0.1136 | 163.1 |
| C: assoc64 | Base RMT n_mem=16 | 0.9887 | 1.0000 | 1.0000 | -0.0119 | 0.0113 | 187.2 |
| C: assoc64 | Base RMT n_mem=8 | 0.9912 | 1.0000 | 1.0000 | -0.0092 | 0.0088 | 177.5 |
| C: assoc64 | FastMem RMT fast_lr=0 slow_freq=2 | 0.9126 | 1.0000 | 1.0000 | -0.0920 | 0.0874 | 178.4 |
| C: assoc64 | FastMem RMT lr=0.005 slow_freq=2 | 0.9176 | 1.0000 | 1.0000 | -0.0868 | 0.0824 | 178.0 |

Interpretation:

- Shape B is a better stress diagnostic than C. It lowers Base RMT n_mem=8 to
  `0.8893` and makes the nonzero FastMem update helpful relative to the
  no-fast control (`0.9191` vs `0.8987` seen accuracy), while still leaving
  Base RMT n_mem=16 strongest (`0.9449`).
- Shape C is not a useful hardening direction for Base RMT under this generator.
  Despite doubling `associations_per_stage`, both Base RMT variants are almost
  solved (`0.9887` and `0.9912` seen accuracy). The extra associations appear
  to change the sampled sequences in a way that reduces the specific early-stage
  retention failure for Base RMT, rather than making the benchmark harder.
- FastMem RMT remains worse than Base RMT on C. The failure is concentrated in
  early stages: with nonzero fast LR, final stage-0/stage-1 accuracies are
  `0.3413/0.1263`, while Base RMT n_mem=8 keeps `0.8497/0.9863`.
- Plasticity is still saturated for all RMT-family models in both shapes. The
  differences are retention/BWT differences, not current-stage learning
  failures.
- The practical next stress branch is B with three seeds, plus maybe a small
  FastMem LR sweep around `0.001`, `0.002`, `0.005`, and `0.01`. C should not be
  escalated unless the data-generation mechanics are changed to make additional
  associations actually increase interference.

### Transformer Online EWC Baseline

Sources:
`results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113711_236162.json`
`results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113722_037701.json`
`results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113735_198965.json`

Setup: formal20 permuted AR, Transformer/MHA only, no replay, stage-local
`OneCycleLR`, `slow_update_mode=accumulate`, seeds 123/456/789. Online EWC
estimates a diagonal Fisher after each stage and accumulates it as
`F <- decay * F + F_stage` with `decay=1.0`; the reference point is reset to the
post-stage parameter snapshot. Fisher is estimated on the current-stage train
loader. Each run saves full metric history, stage-end matrices, current-stage
epoch curves, and EWC Fisher diagnostics.

| EWC Lambda | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 85.6 +/- 0.1 |
| 10 | 0.0344 +/- 0.0223 | 0.8173 +/- 0.1552 | 0.6863 +/- 0.4437 | -0.8241 +/- 0.1400 | 0.7829 +/- 0.1330 | 116.7 +/- 2.1 |
| 100 | 0.0247 +/- 0.0211 | 0.7769 +/- 0.0981 | 0.4443 +/- 0.3947 | -0.7917 +/- 0.0855 | 0.7522 +/- 0.0813 | 117.4 +/- 2.1 |
| 1000 | 0.0663 +/- 0.0175 | 0.9096 +/- 0.0395 | 0.8464 +/- 0.2173 | -0.8877 +/- 0.0232 | 0.8433 +/- 0.0221 | 118.3 +/- 2.0 |
| 10000 | 0.0873 +/- 0.0211 | 0.9710 +/- 0.0213 | 0.8389 +/- 0.2278 | -0.9302 +/- 0.0443 | 0.8837 +/- 0.0421 | 123.8 +/- 3.2 |
| 100000 | 0.2705 +/- 0.1772 | 0.9916 +/- 0.0119 | 1.0000 +/- 0.0000 | -0.7590 +/- 0.1990 | 0.7211 +/- 0.1891 | 119.2 +/- 0.6 |

Interpretation:

- Online EWC is a meaningful non-replay CL comparator, but it does not close
  the gap to RMT variants on formal20 permuted AR. The best EWC setting
  (`lambda=100000`) reaches seen accuracy `0.2705`, far below Base RMT
  (`0.8121`) and FMRMT slow2 (`0.9788`) under the same stage-onecycle setting.
- Low and moderate EWC strengths are mostly ineffective or harmful. They reduce
  forgetting slightly in some seeds but often damage current-stage learning and
  do not materially improve final seen accuracy.
- The high-lambda result is noisy: seed 456 reaches `0.5193` seen accuracy, but
  seeds 123 and 789 are only `0.1203` and `0.1718`. This is a useful baseline,
  not a solved regularization method for this AR stream.
- EWC adds compute overhead from the Fisher pass: Transformer wall time rises
  from about `85.6s` to `119s` per 20-stage run for the full-Fisher settings.
- Paper framing: parameter-importance regularization helps the plain
  Transformer less than architectural/recurrent memory and much less than small
  stored replay. This strengthens the no-replay RMT/FMRMT comparison.

### Transformer Synaptic Intelligence Baseline

Sources:
`results/class_incremental_ar_permuted_si_formal20_permuted_20260524_*.json`

Setup: formal20 permuted AR, Transformer/MHA only, no replay, stage-local
`OneCycleLR`, `slow_update_mode=accumulate`, seeds 123/456/789. Synaptic
Intelligence tracks the per-stage parameter path integral
`sum(-grad * delta_param)`, converts it to a diagonal importance estimate
`path / (delta_param^2 + epsilon)` at each stage boundary, clamps negative
importance to zero, and applies a quadratic penalty to later stages.
Settings: `epsilon=0.1`, `decay=1.0`.

| SI Lambda | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 87.0 +/- 0.5 |
| 100 | 0.3540 +/- 0.2565 | 0.7459 +/- 0.1855 | 0.9013 +/- 0.1327 | -0.4125 +/- 0.1661 | 0.3958 +/- 0.1550 | 130.9 +/- 1.6 |
| 200 | 0.6879 +/- 0.4059 | 0.9307 +/- 0.0981 | 0.7480 +/- 0.3563 | -0.2556 +/- 0.3243 | 0.2444 +/- 0.3102 | 140.6 +/- 3.3 |
| 300 | 0.8594 +/- 0.1157 | 0.9230 +/- 0.0564 | 0.7306 +/- 0.2985 | -0.0670 +/- 0.0909 | 0.0865 +/- 0.1162 | 140.3 +/- 1.7 |
| 500 | 0.5002 +/- 0.1693 | 0.5067 +/- 0.1118 | 0.0000 +/- 0.0000 | -0.0069 +/- 0.0674 | 0.0335 +/- 0.0464 | 141.4 +/- 3.3 |
| 700 | 0.2950 +/- 0.0418 | 0.3042 +/- 0.0415 | 0.0000 +/- 0.0000 | -0.0097 +/- 0.0138 | 0.0152 +/- 0.0128 | 141.3 +/- 2.4 |
| 1000 | 0.1104 +/- 0.0634 | 0.1606 +/- 0.0111 | 0.0000 +/- 0.0000 | -0.0528 +/- 0.0593 | 0.0636 +/- 0.0490 | 139.2 +/- 1.5 |
| 3000 | 0.1239 +/- 0.0119 | 0.1285 +/- 0.0107 | 0.0000 +/- 0.0000 | -0.0048 +/- 0.0019 | 0.0079 +/- 0.0043 | 137.8 +/- 1.5 |
| 10000 | 0.0576 +/- 0.0059 | 0.0785 +/- 0.0106 | 0.0000 +/- 0.0000 | -0.0219 +/- 0.0054 | 0.0208 +/- 0.0051 | 137.4 +/- 0.9 |

Interpretation:

- SI is much stronger than online EWC on formal20 permuted AR. The best SI
  setting here is `lambda=300`, with seen accuracy `0.8594`, BWT `-0.0670`,
  and forgetting from learning `0.0865`.
- This is the first non-replay regularization baseline that beats Base RMT
  retention in this setup (`0.8594` vs Base RMT `0.8121` seen accuracy), but it
  still does not match FMRMT slow2/no-fast slow2 (`~0.978`) or tiny-buffer
  RMT/FMRMT replay.
- SI exposes the retention/plasticity tradeoff clearly. `lambda=300` preserves
  old stages well but late-stage learning degrades: mean learning accuracy for
  stages 15-19 is approximately `0.89, 0.73, 0.53, 0.59, 0.73`. Higher lambdas
  reduce forgetting mostly by preventing new learning; at `lambda>=500`, final
  plasticity is zero.
- Paper framing: SI is a strong and necessary no-replay CL comparator. It
  narrows the gap to RMT variants substantially, but its best point still shows
  late-stage intransigence/plasticity loss, while the current FMRMT/RMT
  stage-onecycle settings retain perfect current-stage learning.

### Formal 20-Stage Permuted AR Core Comparison

Sources:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_core_formal20_permuted_formal20_permuted_20260525_0409*.json`
`results/class_incremental_ar_permuted_online_ewc_core_formal20_permuted_20260525_0411*.json`
`results/class_incremental_ar_permuted_si_core_formal20_permuted_20260525_0414*.json`

Setup: formal20 permuted AR, stage-local `OneCycleLR`,
`slow_update_mode=accumulate`, seeds 123/456/789, no replay/buffered old
examples. RMT-family runs use `segment_len=64`; matched Base RMT/FastMem RMT
comparison uses `n_mem=8`. Base RMT `n_mem=16` is reported as a capacity
reference.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 | 84.9 +/- 0.3 |
| Transformer + online EWC, lambda=100000 | 0.2705 +/- 0.1772 | 0.9916 +/- 0.0119 | 1.0000 +/- 0.0000 | -0.7590 +/- 0.1990 | 0.7211 +/- 0.1891 | 115.8 +/- 1.7 |
| Transformer + SI, lambda=300 | 0.8594 +/- 0.1157 | 0.9230 +/- 0.0564 | 0.7306 +/- 0.2985 | -0.0670 +/- 0.0909 | 0.0865 +/- 0.1162 | 132.7 +/- 0.5 |
| Base RMT, n_mem=8 | 0.8718 +/- 0.0719 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1349 +/- 0.0757 | 0.1282 +/- 0.0719 | 119.5 +/- 0.2 |
| Base RMT, n_mem=16 | 0.8121 +/- 0.0120 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1978 +/- 0.0126 | 0.1879 +/- 0.0120 | 124.3 +/- 0.2 |
| FastMem RMT, fast_lr=0, slow_freq=2 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 | 123.6 +/- 0.2 |
| FastMem RMT, fast_lr=0.005, slow_freq=2 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 | 124.3 +/- 0.1 |

Interpretation:

- The matched `n_mem=8` comparison is now complete. FastMem RMT has the best
  retention and BWT in the core setting, while preserving perfect
  current-stage learning.
- Base RMT `n_mem=8` is a fairer matched-memory baseline than Base RMT
  `n_mem=16`, and it is stronger here (`0.8718` vs `0.8121` seen accuracy).
  The `n_mem=16` result should be framed as a capacity reference, not the
  primary matched comparison.
- SI remains the strongest Transformer-only continual-learning baseline. It is
  competitive with Base RMT on final seen accuracy, but it pays a plasticity
  cost: final plasticity is `0.7306 +/- 0.2985`, with strong seed variability.
- Online EWC improves over plain Transformer but remains far below SI and the
  RMT-family models.
- The nonzero fast-memory update is not isolated as the causal reason for the
  formal20 result: `fast_lr=0.005` and `fast_lr=0` are essentially tied. The
  defensible claim is that the FastMem RMT architecture/slow-update recipe is
  strongest in this core online setting; the local fast update needs a harder
  diagnostic to show clear benefit.

### Balanced Oracle Replay Budget Sweep

Sources:
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay4_formal20_permuted_20260524_054908_476552.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay4_formal20_permuted_20260524_054908_677505.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay4_formal20_permuted_20260524_054910_470293.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay8_formal20_permuted_20260524_055505_488207.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay8_formal20_permuted_20260524_055506_275663.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay8_formal20_permuted_20260524_055507_949277.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay16_formal20_permuted_20260524_060101_139254.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay16_formal20_permuted_20260524_060102_145768.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay16_formal20_permuted_20260524_060103_473541.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay32_formal20_permuted_20260524_060705_280352.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay32_formal20_permuted_20260524_060708_727551.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay32_formal20_permuted_20260524_060708_911634.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay64_formal20_permuted_20260524_053209_137266.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay64_formal20_permuted_20260524_053212_262523.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_replay64_formal20_permuted_20260524_053213_598562.json`

Setup: formal20 permuted AR, `stage_onecycle`, `slow_update_mode=accumulate`,
fixed total train budget, replay examples per old stage in
`{4, 8, 16, 32, 64}`, seeds 123/456/789. At stage `t`, the stage train set has
`budget * t` old examples and `2048 - budget * t` current examples per epoch,
keeping 2048 total examples per epoch. Important caveat: this runner resamples
old-stage examples from the synthetic generator at each later stage. It is
therefore best interpreted as oracle/generative balanced replay, not a stored
fixed-buffer experience replay baseline.

| Model | Replay/Old Stage | Current/Old Examples at Stage 19 | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA + replay | 4 | 1972 / 76 | 0.3797 +/- 0.3929 | 0.8959 +/- 0.0707 | 1.0000 +/- 0.0000 | -0.5434 +/- 0.3752 | 0.5230 +/- 0.3471 | 76.6 +/- 0.2 |
| Transformer/MHA + replay | 8 | 1896 / 152 | 0.9910 +/- 0.0043 | 0.9861 +/- 0.0173 | 1.0000 +/- 0.0000 | 0.0052 +/- 0.0175 | 0.0079 +/- 0.0041 | 76.8 +/- 0.3 |
| Transformer/MHA + replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 76.7 +/- 0.2 |
| Transformer/MHA + replay | 32 | 1440 / 608 | 1.0000 +/- 0.0000 | 0.9709 +/- 0.0412 | 1.0000 +/- 0.0000 | 0.0307 +/- 0.0433 | 0.0000 +/- 0.0000 | 76.7 +/- 0.1 |
| Transformer/MHA + replay | 64 | 832 / 1216 | 1.0000 +/- 0.0000 | 0.9843 +/- 0.0221 | 1.0000 +/- 0.0000 | 0.0165 +/- 0.0233 | 0.0000 +/- 0.0000 | 76.9 +/- 0.2 |
| Base RMT n_mem=16 + replay | 4 | 1972 / 76 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.7 +/- 0.3 |
| Base RMT n_mem=16 + replay | 8 | 1896 / 152 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 116.0 +/- 0.3 |
| Base RMT n_mem=16 + replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.7 +/- 0.4 |
| Base RMT n_mem=16 + replay | 32 | 1440 / 608 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.4 +/- 0.3 |
| Base RMT n_mem=16 + replay | 64 | 832 / 1216 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 116.2 +/- 0.5 |
| FMRMT lr=0.005 slow_freq=2 + replay | 4 | 1972 / 76 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 116.1 +/- 0.6 |
| FMRMT lr=0.005 slow_freq=2 + replay | 8 | 1896 / 152 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 117.0 +/- 1.0 |
| FMRMT lr=0.005 slow_freq=2 + replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 116.2 +/- 0.9 |
| FMRMT lr=0.005 slow_freq=2 + replay | 32 | 1440 / 608 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.1 +/- 0.3 |
| FMRMT lr=0.005 slow_freq=2 + replay | 64 | 832 / 1216 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 116.1 +/- 0.5 |

Interpretation:

- Oracle balanced replay is an extremely strong CL baseline on formal20
  permuted AR. Base RMT and FMRMT saturate with only 4 generated old examples
  per old stage, i.e. only 76 old examples total at stage 19 under the fixed
  2048-example stage budget.
- Transformer/MHA is much more replay-sensitive. Replay4 is unstable across
  seeds, but replay8 already reaches `0.9910` final seen accuracy and replay16
  fully saturates.
- This changes the paper framing, but with a caveat. The no-replay RMT/FMRMT
  result is meaningful as a no-buffer continual-learning result; the oracle
  replay result shows that the task is easy to stabilize if old-stage samples
  remain available.
- Before using this as the main standard CL baseline, run a true stored-buffer
  replay control. The stored-buffer question is whether a small fixed set of old
  examples, rather than fresh old-stage resampling, still saturates the task.

### Stored-Buffer Replay Budget Sweep

Sources:
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay4_formal20_permuted_20260524_063405_723755.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay4_formal20_permuted_20260524_063406_354318.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay4_formal20_permuted_20260524_063409_558854.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay1_formal20_permuted_20260524_070048_329982.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay1_formal20_permuted_20260524_070049_076588.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay1_formal20_permuted_20260524_070053_573651.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay2_formal20_permuted_20260524_070645_818210.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay2_formal20_permuted_20260524_070648_343591.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay2_formal20_permuted_20260524_070651_018544.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay8_formal20_permuted_20260524_064001_016782.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay8_formal20_permuted_20260524_064003_866973.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay8_formal20_permuted_20260524_064004_037528.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay16_formal20_permuted_20260524_064554_526596.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay16_formal20_permuted_20260524_064555_745663.json`
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay16_formal20_permuted_20260524_064558_131325.json`

Setup: same formal20 permuted AR and fixed-total replay budget as above, but
old examples are true stored-buffer examples selected from each stage's first
current-stage train stream. They are not regenerated at later stages.

| Model | Stored Replay/Old Stage | Current/Old Examples at Stage 19 | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA + stored replay | 1 | 2029 / 19 | 0.0759 +/- 0.0375 | 0.8224 +/- 0.1264 | 0.9709 +/- 0.0411 | -0.7858 +/- 0.0940 | 0.7465 +/- 0.0893 | 76.3 +/- 0.2 |
| Transformer/MHA + stored replay | 2 | 2010 / 38 | 0.2765 +/- 0.3322 | 0.8558 +/- 0.1690 | 0.6763 +/- 0.4342 | -0.6098 +/- 0.2665 | 0.5793 +/- 0.2532 | 77.1 +/- 0.3 |
| Transformer/MHA + stored replay | 4 | 1972 / 76 | 0.1237 +/- 0.1213 | 0.6772 +/- 0.2351 | 0.5506 +/- 0.3379 | -0.5826 +/- 0.1296 | 0.5535 +/- 0.1231 | 76.7 +/- 0.5 |
| Transformer/MHA + stored replay | 8 | 1896 / 152 | 0.9312 +/- 0.0926 | 0.8941 +/- 0.0572 | 1.0000 +/- 0.0000 | 0.0391 +/- 0.1080 | 0.0415 +/- 0.0564 | 77.0 +/- 0.3 |
| Transformer/MHA + stored replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 0.9722 +/- 0.0393 | 1.0000 +/- 0.0000 | 0.0292 +/- 0.0413 | 0.0000 +/- 0.0000 | 77.4 +/- 0.2 |
| Base RMT n_mem=16 + stored replay | 1 | 2029 / 19 | 0.9502 +/- 0.0091 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0524 +/- 0.0095 | 0.0498 +/- 0.0091 | 115.2 +/- 0.4 |
| Base RMT n_mem=16 + stored replay | 2 | 2010 / 38 | 0.9712 +/- 0.0108 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0303 +/- 0.0114 | 0.0288 +/- 0.0108 | 116.2 +/- 0.4 |
| Base RMT n_mem=16 + stored replay | 4 | 1972 / 76 | 0.9945 +/- 0.0018 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0058 +/- 0.0019 | 0.0055 +/- 0.0018 | 115.8 +/- 0.6 |
| Base RMT n_mem=16 + stored replay | 8 | 1896 / 152 | 0.9989 +/- 0.0016 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0012 +/- 0.0016 | 0.0011 +/- 0.0016 | 115.8 +/- 0.6 |
| Base RMT n_mem=16 + stored replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.6 +/- 0.6 |
| FMRMT lr=0.005 slow_freq=2 + stored replay | 1 | 2029 / 19 | 0.9874 +/- 0.0128 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0133 +/- 0.0135 | 0.0126 +/- 0.0128 | 115.4 +/- 0.6 |
| FMRMT lr=0.005 slow_freq=2 + stored replay | 2 | 2010 / 38 | 0.9983 +/- 0.0012 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0017 +/- 0.0013 | 0.0017 +/- 0.0012 | 116.4 +/- 0.8 |
| FMRMT lr=0.005 slow_freq=2 + stored replay | 4 | 1972 / 76 | 0.9989 +/- 0.0016 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0012 +/- 0.0016 | 0.0011 +/- 0.0016 | 115.2 +/- 0.7 |
| FMRMT lr=0.005 slow_freq=2 + stored replay | 8 | 1896 / 152 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.7 +/- 0.5 |
| FMRMT lr=0.005 slow_freq=2 + stored replay | 16 | 1744 / 304 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 115.4 +/- 1.0 |

Interpretation:

- True stored-buffer replay is still a very strong CL baseline, but it is less
  forgiving than oracle replay for Transformer/MHA. Transformer remains
  unstable through replay4, becomes strong but variable at replay8, and fully
  retains all stages at replay16.
- Base RMT and FMRMT are much more replay-efficient than Transformer. Both keep
  perfect current-stage learning even with one stored example per old stage, but
  retention differs: FMRMT reaches `0.9874` seen accuracy at replay1 and
  `0.9983` at replay2, while Base RMT reaches `0.9502` and `0.9712`.
- The tiny-buffer regime is the clearest FMRMT-over-Base replay result so far.
  FMRMT needs only 2 stored examples per old stage to roughly match its
  replay4/replay8 performance, while Base RMT needs about 4-8.
- Stored replay clarifies the paper framing: RMT-style memory is valuable in
  no-buffer and tiny-buffer regimes, while ordinary replay can solve formal20
  permuted AR once enough old examples are stored.

### Reservoir Replay Budget Sweep

Sources:
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_reservoir_replay*_buffer*_formal20_permuted_20260524_*.json`

Setup: formal20 permuted AR, Transformer/MHA only, fixed-total train budget,
stage-onecycle scheduler, slow-update accumulation setting carried for
consistency, three seeds. Unlike the balanced stored-buffer sweep above, old
examples are sampled from a single fixed-capacity reservoir over the observed
training stream. The default reservoir capacity is
`replay_examples_per_old_stage * 19`, matching the final memory size of the
balanced stored replay setting.

| Model | Reservoir Replay/Old Stage | Reservoir Capacity | Current/Old Examples at Stage 19 | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA + reservoir replay | 1 | 19 | 2029 / 19 | 0.0535 +/- 0.0216 | 0.7833 +/- 0.1623 | 0.7870 +/- 0.3012 | -0.7682 +/- 0.1753 | 0.7298 +/- 0.1666 | 77.1 +/- 0.3 |
| Transformer/MHA + reservoir replay | 2 | 38 | 2010 / 38 | 0.1470 +/- 0.0845 | 0.7431 +/- 0.2070 | 1.0000 +/- 0.0000 | -0.6274 +/- 0.1388 | 0.5961 +/- 0.1318 | 77.1 +/- 0.1 |
| Transformer/MHA + reservoir replay | 4 | 76 | 1972 / 76 | 0.6646 +/- 0.2260 | 0.8131 +/- 0.1327 | 1.0000 +/- 0.0000 | -0.1563 +/- 0.0984 | 0.1671 +/- 0.0873 | 77.3 +/- 0.4 |
| Transformer/MHA + reservoir replay | 8 | 152 | 1896 / 152 | 0.8388 +/- 0.1984 | 0.9503 +/- 0.0550 | 1.0000 +/- 0.0000 | -0.1173 +/- 0.1519 | 0.1162 +/- 0.1408 | 80.1 +/- 0.4 |
| Transformer/MHA + reservoir replay | 16 | 304 | 1744 / 304 | 0.9999 +/- 0.0001 | 0.9907 +/- 0.0131 | 1.0000 +/- 0.0000 | 0.0097 +/- 0.0137 | 0.0001 +/- 0.0001 | 77.1 +/- 0.2 |

Interpretation:

- Reservoir replay is the standard fixed-buffer ER comparison that was missing
  from the earlier balanced-replay result. It is harder than oracle replay and
  less controlled than balanced stored replay because the buffer is not
  stage-stratified.
- Transformer/MHA with reservoir replay still needs a moderate buffer to solve
  formal20 permuted AR. Tiny reservoirs remain weak: capacities 19 and 38 give
  only `0.0535` and `0.1470` final seen accuracy. Capacity 152 is strong but
  still variable; capacity 304 solves the task.
- This strengthens, rather than weakens, the no-buffer/tiny-buffer RMT result.
  No-replay FMRMT slow2 reaches about `0.9788` seen accuracy, and Base RMT
  reaches about `0.8121`; Transformer/MHA needs reservoir capacity 152-304 to
  enter the same retention range.
- Balanced stored replay remains useful as a favorable, stage-stratified replay
  control. Reservoir replay should be the main "standard ER" baseline in the
  paper, with balanced replay described as a stronger controlled replay variant.

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

### Budget16 Raw JSON Rerun

Sources:
`results/interference_ar_calibrated_budget16_raw_continual_latest_fixed_all8_budget16_formal_cl_20260524_110450_335079.json`
`results/interference_ar_calibrated_budget16_raw_continual_latest_fixed_all8_budget16_formal_cl_20260524_103822.json`
`results/interference_ar_calibrated_budget16_raw_continual_latest_fixed_all8_budget16_formal_cl_20260524_103823.json`

Supersedes reconstructed source:
`results/interference_ar_calibrated_continual_latest_fixed_all8_budget16_comet_reconstructed_20260520_145536.json`

The local raw JSON runner now stores each run's config dump and full logger
history. Each model/seed has 5209 logged metric records, so this artifact is
usable for stage-matrix and per-epoch diagnostics. Seed 123 was rerun after
fixing a second-level timestamp collision in the output filename; its metrics
match the original terminal log.

Setup difference from the low-budget run above: 4096 train examples/stage, 1024
test examples/stage, 16 epochs/stage, same 5-stage fixed-shift all-updated
latest-value task and seeds 123, 456, 789.

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning | Total Wall Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.1751 +/- 0.0485 | 0.9510 +/- 0.0316 | 0.7550 +/- 0.1581 | -0.9698 +/- 0.0213 | 0.7759 +/- 0.0171 | 66.9 +/- 1.0 |
| Base RMT n_mem=16 | 0.7972 +/- 0.0070 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.2535 +/- 0.0087 | 0.2028 +/- 0.0070 | 102.8 +/- 1.9 |
| FMRMT stable lr=0.005 slow_freq=4 | 0.8678 +/- 0.0647 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1652 +/- 0.0809 | 0.1322 +/- 0.0647 | 102.1 +/- 1.9 |
| FMRMT plastic lr=0.01 slow_freq=1 | 0.7474 +/- 0.0375 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.3157 +/- 0.0469 | 0.2526 +/- 0.0375 | 105.7 +/- 2.5 |

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
- The raw rerun confirms the reconstructed Comet/stdout result; the small
  numeric changes are from using the local JSON aggregator's population standard
  deviation convention.
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

- Add a compute-matched Transformer baseline if we want a stronger fairness
  claim than "same Zoology default width/depth".
- Add old-vs-new class accuracy to separate plasticity from retention more
  directly.
- Repeat the stable FMRMT default on harder variants: more stages, more
  associations per stage, and longer distractor spans.
- Redesign vocab-growth MQAR into an easier calibration ladder before using it
  for ranking: fewer query pairs, smaller value ranges, more train examples, or
  fixed per-stage associations instead of fully episodic random values.
