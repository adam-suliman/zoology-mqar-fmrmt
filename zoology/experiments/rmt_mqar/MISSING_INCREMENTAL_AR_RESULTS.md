# Missing Incremental AR Results for Paper

This file tracks only the class-incremental associative-retrieval experiments
that are still needed before the FastMem-RMT paper can make strong claims about
incremental AR. It is intentionally separate from `RESULTS.md`: this is a run
queue and claim-audit checklist, not a result summary.

## Ground Rules

- Use the current formal continual-learning protocol with stage-local test
  splits. Do not use old cumulative-prefix artifacts for paper claims.
- Main incremental-AR scope is the standard online stream without storing old
  examples. Experiments that add stored, generated, or sampled old examples
  should be explicitly labeled as buffered/replay experiments.
- Save raw JSON under `results/` for every substantial run.
- Use seeds `123,456,789` for main tables unless an experiment is explicitly
  marked as a smoke or diagnostic.
- Report at least:
  - `continual/plasticity`
  - `continual/avg_learning_accuracy`
  - `continual/seen_avg_accuracy`
  - `continual/avg_bwt`
  - `continual/avg_forgetting_from_learning`
  - stage-level train/eval wall seconds and throughput
- Interpret low current-stage accuracy as plasticity or learning failure.
  Interpret old-stage degradation after learning as forgetting / negative BWT.
- For fair RMT-family comparisons, use matched-memory controls as the primary
  comparison: Base RMT `n_mem=8`, FastMem RMT `n_mem=8`, and FastMem RMT
  `fast_lr=0` with the same `n_mem`, `segment_len`, schedule, train budget,
  and seeds. Base RMT `n_mem=16` should be reported separately as a stronger
  capacity/best-baseline reference, not as the capacity-matched baseline.

## Recently Completed Controls

These are no longer missing and should be treated as current paper-relevant
context:

- Formal 20-stage permuted AR with fixed per-stage value permutations,
  `association_table_seed=20260522`.
- Scheduler control on formal20 permuted AR:
  `global_cosine`, `stage_cosine`, and `constant`.
- CIFAR-style slow-gradient accumulation control for formal20 permuted AR.
- Stage-local `OneCycleLR` control with scheduler stepping on slow optimizer
  steps.
- Slow-frequency and no-fast attribution controls for the current strongest
  setting:
  - FMRMT `slow_freq=2`, `rmt_fast_lr=0.005`
  - FMRMT `slow_freq=2`, `rmt_fast_lr=0.0`
  - FMRMT `slow_freq=4`, `rmt_fast_lr=0.005`
  - FMRMT `slow_freq=4`, `rmt_fast_lr=0.0`
- Fast-LR sweep at `slow_freq=2`, including
  `rmt_fast_lr in {0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}`.
- Oracle/generative balanced replay with fixed-total budgets
  `replay_examples_per_old_stage in {4, 8, 16, 32, 64}` on formal20 permuted AR
  for Transformer/MHA, Base RMT `n_mem=16`, and FMRMT `slow_freq=2`,
  `rmt_fast_lr=0.005`.
- Stored-buffer replay with fixed-total budgets
  `replay_examples_per_old_stage in {1, 2, 4, 8, 16}` on formal20 permuted AR
  for the same models.
- Reservoir replay with fixed-total budgets
  `replay_examples_per_old_stage in {1, 2, 4, 8, 16}` on formal20 permuted AR
  for Transformer/MHA. Reservoir capacity is
  `replay_examples_per_old_stage * 19`, matching the final memory size of the
  balanced stored-buffer setting.
- Raw JSON rerun of Budget16 calibrated fixed-shift all-updated interference
  for Transformer/MHA, Base RMT `n_mem=16`, stable FMRMT, and plastic FMRMT.
- Online EWC runner for formal permuted AR, with a passing 3-stage smoke test.
- Formal20 online EWC lambda sweep for Transformer/MHA with
  `lambda in {0, 10, 100, 1000, 10000, 100000}` and seeds `123,456,789`.
- Synaptic Intelligence runner for formal permuted AR, with a passing 3-stage
  smoke test and formal20 lambda sweep over
  `{0, 100, 200, 300, 500, 700, 1000, 3000, 10000}`.
- Segment-length control with `segment_len=128` on formal20 permuted AR for
  Transformer/MHA, Base RMT `n_mem=16`, FMRMT `rmt_fast_lr=0.005, slow_freq=2`,
  and FMRMT `rmt_fast_lr=0.0, slow_freq=2`.
- Raw JSON now includes full metric history, per-stage matrices, and
  per-epoch current-stage learning curves for these newer runs.

Current interpretation from these controls:

- The late zero-plasticity failure under 20-stage `global_cosine` was a
  scheduler confound, not evidence of intrinsic model plasticity loss.
- `stage_onecycle + slow_update_mode=accumulate + slow_freq=2` gives the best
  20-stage permuted AR retention/plasticity tradeoff among current RMT
  variants.
- `rmt_fast_lr=0.005` is the best nonzero fast LR in that recipe, but the
  no-fast control is nearly tied. Do not claim the local fast-memory gradient
  update is the isolated cause of the 20-stage AR gain.
- Replay is out of the main incremental-AR scope for now. Existing replay runs
  show that the task can be solved with enough stored examples, which makes them
  less useful for the core online story. Keep these results as appendix
  context only if needed.
- SI is a substantially stronger no-replay regularization baseline than EWC.
  Its best current setting, `lambda=300`, reaches seen accuracy `0.8594` and
  BWT `-0.0670`, beating Base RMT retention but still below FMRMT/slow-update
  RMT. Higher SI lambdas reduce forgetting by causing plasticity failure.
- Segment_len=128 confirms that cross-segment recurrence is not required for
  the broad RMT-vs-Transformer gap, but it helps the strongest FMRMT setting.
  FMRMT lr=0.005 slow2 drops from `0.9788` to `0.9144` seen accuracy when the
  two recurrent segments are collapsed into one segment.
- Matched core comparison is complete for formal20 permuted AR with
  `stage_onecycle`, `slow_update_mode=accumulate`, and seeds `123,456,789`.
  Raw JSON:
  `results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_core_formal20_permuted_formal20_permuted_20260525_0409*.json`,
  `results/class_incremental_ar_permuted_online_ewc_core_formal20_permuted_20260525_0411*.json`,
  and
  `results/class_incremental_ar_permuted_si_core_formal20_permuted_20260525_0414*.json`.

  | Model | Seen avg acc | Learning acc | Plasticity | Avg BWT | Forgetting from learning |
  |---|---:|---:|---:|---:|---:|
  | Transformer/MHA | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 |
  | Transformer + online EWC, lambda=100000 | 0.2705 +/- 0.1772 | 0.9916 +/- 0.0119 | 1.0000 +/- 0.0000 | -0.7590 +/- 0.1990 | 0.7211 +/- 0.1891 |
  | Transformer + SI, lambda=300 | 0.8594 +/- 0.1157 | 0.9230 +/- 0.0564 | 0.7306 +/- 0.2985 | -0.0670 +/- 0.0909 | 0.0865 +/- 0.1162 |
  | Base RMT, n_mem=8 | 0.8718 +/- 0.0719 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1349 +/- 0.0757 | 0.1282 +/- 0.0719 |
  | Base RMT, n_mem=16 | 0.8121 +/- 0.0120 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1978 +/- 0.0126 | 0.1879 +/- 0.0120 |
  | FastMem RMT, fast_lr=0, slow_freq=2 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 |
  | FastMem RMT, fast_lr=0.005, slow_freq=2 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 |

## Highest-Priority Missing Experiments

### 1. Matched Core RMT-Family Table

Current status: complete.

Required for the narrowed scope:

- Formal20 permuted AR, stage-local `OneCycleLR`, `slow_update_mode=accumulate`,
  seeds `123,456,789`.
- Same task, train budget, segment length, and scheduler for all models.
- Primary matched models:
  - Transformer/MHA
  - Base RMT `n_mem=8`
  - FastMem RMT `n_mem=8`, `rmt_fast_lr=0.005`, `slow_freq=2`
  - FastMem RMT `n_mem=8`, `rmt_fast_lr=0.0`, `slow_freq=2`
- Secondary capacity reference:
  - Base RMT `n_mem=16`

Reason:

- The older headline table uses Base RMT `n_mem=16` and FastMem RMT `n_mem=8`.
  That is fine as a strong-baseline reference, but the paper needs the matched
  `n_mem=8` table to avoid overclaiming.
- The matched table keeps the broad result: FastMem RMT has the best retention,
  but the `fast_lr=0` control is essentially tied. This supports the online
  memory architecture/slow-update story more strongly than a causal claim about
  the local fast-memory gradient update.

### 2. Robustness Metrics for Core Runs

Current status: missing.

Add and report:

- current-stage test loss / NLL at learning time;
- correct-token probability on masked query positions;
- logit margin between the correct value token and the strongest wrong value;
- optionally per-key accuracy.

Reason:

- Top-1 current-stage accuracy can jump from `0` to `1` while loss decreases
  smoothly. Accuracy is valid for BWT/forgetting, but too coarse to claim robust
  consolidation.

### 3. Replay Appendix Only

Current status: complete enough; do not prioritize.

Existing replay results:

- Oracle/generative balanced replay.
- Stored fixed-buffer replay.
- Reservoir replay.

Use only as appendix/context unless the paper scope changes back. Replay shows
that formal20 permuted AR is solvable with enough old examples, which weakens
its usefulness as the main online memory comparison.

Optional only:

- Replay-overhead run after the fixed-total sweep, if we want to show the
  upper-bound behavior when replay adds examples instead of replacing
  current-stage examples.

### 4. Formal Permuted-Mapping Clean AR Calibration

Current status: 20-stage formal permuted runs exist; 5-stage and 10-stage
permuted calibration tables are still optional/missing.

Needed runs:

- 5-stage clean class-incremental AR with `value_mapping="permuted"` and fixed
  `association_table_seed=20260522`.
- 10-stage horizon with the same permuted mapping.
- Models:
  - Transformer/MHA
  - Base RMT `n_mem=4` for 5-stage
  - Base RMT `n_mem=16` for 10-stage
  - stable FMRMT `n_mem=8, rmt_fast_lr=0.005, slow_freq=4`
  - tuned/plastic FMRMT `n_mem=8, rmt_fast_lr=0.01, slow_freq=1`

Purpose:

- Removes the old offset-aligned key/value shortcut concern.
- Gives the paper a cleaner controlled AR benchmark.
- Helps decide whether the paper should report a 20-stage table only or also
  include a shorter 5/10-stage calibration ladder.

### 5. Fast-Memory Causality Controls

Current status: partially complete on formal20 permuted AR.

Needed runs:

- Optional FMRMT architecture with the same fast-memory settings but
  `slow_freq=1` under `stage_onecycle + accumulate`.
- Base RMT with matched `n_mem=8` where possible.
- Any 5-stage/10-stage versions needed for appendix consistency.

Purpose:

- Separates the effect of gradient-updated fast memory from the effect of
  slower optimizer cadence.
- Prevents overclaiming that all retention gains come from fast memory itself.
- This is the most important ablation for the paper's mechanism claim.

### 6. Optimizer-Step / Compute-Matched AR Controls

Current status: accumulation and timing instrumentation exist for formal20;
wall-clock or exact slow-step matching is still optional/missing.

Needed runs:

- Match recurrent baselines by one or more of:
  - same examples/tokens
  - same wall-clock budget
  - same number of slow optimizer steps
- Include timing metrics in the raw JSON.

Purpose:

- The local AR trainer skips slow optimizer steps when `slow_freq > 1`; it does
  not accumulate all skipped gradients.
- A retention gain under `slow_freq=4` may partly reflect fewer slow parameter
  updates. This must be controlled before making a strong mechanism claim.

### 7. Raw JSON Rerun of Budget16 Calibrated Interference

Current status: complete.

Completed run:

- 5-stage fixed-shift all-updated latest-value interference.
- Budget: 4096 train examples/stage, 1024 test examples/stage,
  16 epochs/stage.
- Seeds: `123,456,789`.
- Models:
  - Transformer/MHA
  - Base RMT `n_mem=16`
  - stable FMRMT
  - tuned/plastic FMRMT

Purpose:

- Produces a clean raw artifact for the paper's overwrite/interference table.
- Avoids relying on reconstructed Comet output.

Result:

- Confirms the reconstructed Comet/stdout numbers with raw JSON histories.
- All RMT variants retain perfect current-stage learning at this budget.
- Stable FMRMT has the best retention/BWT; plastic FMRMT forgets more than Base
  RMT.

### 8. Segment-Length Control

Current status: complete for formal20 permuted AR.

Completed runs:

- Formal20 permuted AR with `segment_len=128`.
- Models:
  - Transformer/MHA
  - Base RMT `n_mem=16`
  - FMRMT `rmt_fast_lr=0.005, slow_freq=2`
  - FMRMT `rmt_fast_lr=0.0, slow_freq=2`

Result:

- Transformer is unchanged, as expected.
- Base RMT remains strong but slightly worse/noisier: `0.8121 -> 0.7920` seen
  accuracy.
- FMRMT lr=0.005 slow2 drops from `0.9788 -> 0.9144` seen accuracy.
- FMRMT fast0 slow2 drops from `0.9774 -> 0.9432` seen accuracy.
- Cross-segment recurrence helps the best FMRMT setting, but the remaining
  segment_len=128 gain over Transformer means the RMT memory-token architecture
  and slow-update recipe matter even without recurrence across sample segments.

Purpose:

- Checks whether RMT/FMRMT gains depend on actual recurrence across two
  sequence segments or mainly on the memory-augmented block itself.
- Important for any claim about recurrent memory rather than simply extra
  memory tokens / architectural bias.

## Main Paper Table Runs Still Needed

### 9. Clean AR Paper Table

Recommended setup:

- Prefer the formal permuted-mapping variant if it passes calibration.
- Otherwise report the existing aligned formal AR result with a clear caveat.
- Include one-stage solvability in appendix or text.

Models:

- Transformer/MHA
- stronger Transformer if implemented
- Base RMT best memory size for the task
- stable FMRMT
- tuned/plastic FMRMT
- selected CL baselines from the list below, if implemented

### 10. Horizon Table With Tuned FMRMT

Current status:

- Existing horizon table has stable FMRMT.
- 10-stage FMRMT tuning exists.
- Strongest Base RMT comparator for 10-stage is `n_mem=16`.

Needed runs:

- 10-stage clean/permuted horizon with:
  - Base RMT `n_mem=16`
  - stable FMRMT
  - tuned/plastic FMRMT
  - Transformer/MHA
- Optional 20-stage confirmation with the same model set.

Purpose:

- Shows whether tuned FMRMT restores late-stage plasticity without losing too
  much retention.
- Prevents comparing stable FMRMT only against a weaker Base RMT setting.

### 11. Interference Table With Solvable One-Stage Calibration

Current status:

- Random repeated-key latest-value is not cleanly solved in one-stage form.
- Fixed-shift all-updated latest-value passes one-stage calibration.

Needed runs:

- Use fixed-shift all-updated latest-value as the paper interference benchmark.
- Report one-stage calibration.
- Report 5-stage low-budget and Budget16 raw JSON runs if both are available.

Purpose:

- Gives a controlled overwrite/interference task where low current-stage
  accuracy is not caused by basic one-stage unsolvability.

## Optional But Valuable Diagnostics

### 12. Adapted-Memory vs Initial-Memory Evaluation

Needed runs or instrumentation:

- Evaluate local FMRMT with:
  - default eval memory policy
  - zero/initial memory
  - carried/adapted training memory where meaningful

Purpose:

- Clarifies whether paper gains come from learned slow memory initializer,
  training-time fast state, or both.
- Important because local NLP FMRMT evaluates with `initial_memory_tokens`, not
  the mutable `fast_memory_tokens`.

### 13. Memory Diagnostics

Needed logs:

- memory norm
- memory update norm
- memory gradient norm
- per-stage accuracy matrix heatmaps
- current-stage learning curves
- old-stage retention curves

Purpose:

- Supports mechanism discussion without relying only on final scalar metrics.

## Continual-Learning Methods Applicable to This Pipeline

These methods should be feasible because the AR pipeline already has staged
datasets, stage-local evaluation, and per-stage train loops.

### Replay-Based

- Experience Replay with a fixed-size buffer: implemented as reservoir replay
  for Transformer/MHA.
- Reservoir Replay: complete for formal20 permuted AR.
- Balanced per-stage replay: complete for oracle and stored-buffer variants.
- DER / DER++ style replay with stored logits, if logits are easy to persist.
- Cumulative replay / joint training as an upper bound, not as a strict CL
  method.

Why useful:

- Strong baseline for forgetting/BWT.
- Easy to explain and likely strong on synthetic AR.

### Regularization-Based

- EWC / online EWC.
- Synaptic Intelligence.
- MAS.

Why useful:

- Tests whether parameter-importance regularization can preserve old
  associations without external memory.
- Provides standard catastrophic-forgetting baselines.

Current status:

- Online EWC is implemented in
  `zoology/experiments/rmt_mqar/class_incremental_ar_permuted_ewc.py`.
- Synaptic Intelligence is implemented in
  `zoology/experiments/rmt_mqar/class_incremental_ar_permuted_si.py`.
- Smoke result:
  `results/class_incremental_ar_permuted_online_ewc_ewc3_permuted_smoke_20260524_111921_386611.json`.
- Full formal20 sweep is complete:
  - `results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113711_236162.json`
  - `results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113722_037701.json`
  - `results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_113735_198965.json`
- Result: best lambda is `100000`, but it is noisy and reaches only
  `0.2705` seen accuracy, far below Base RMT and FMRMT.
- SI formal20 sweep is complete:
  - broad sweep:
    `results/class_incremental_ar_permuted_si_formal20_permuted_20260524_*.json`
  - refined sweep:
    `results/class_incremental_ar_permuted_si_formal20_permuted_20260524_*.json`
- Result: best lambda is `300`, reaching `0.8594` seen accuracy, `0.9230`
  average learning accuracy, BWT `-0.0670`, and forgetting from learning
  `0.0865`.

### Distillation-Based

- Learning without Forgetting.
- Logit distillation on old-stage probes.

Why useful:

- Natural fit if we keep a frozen teacher after each stage.
- Can be combined with replay-lite probe batches.

### Gradient-Constraint Replay

- GEM.
- A-GEM.

Why useful:

- Directly targets negative transfer on replayed old examples.
- More complex than plain replay, so should come after simple replay baselines.

### Plasticity-Preserving / Reset Methods

- Shrink-and-perturb.
- ReDo.
- Continual Backpropagation.
- Selective Weight Reinitialization.
- Stage-boundary reset or partial reset baselines.

Why useful:

- Relevant to the paper's plasticity-loss framing.
- Best used on harder/horizon AR once one-stage solvability and scheduler
  confounds are controlled.

## Recommended Order

1. Run 5-stage/10-stage permuted calibration tables only if the paper needs a
   shorter calibration ladder in addition to formal20.
2. Only then add heavier methods such as DER, A-GEM, ReDo, CBP, or SWR.
3. Treat replay-overhead or larger replay budgets as appendix-only unless the
   main paper needs a replay upper-bound curve.
