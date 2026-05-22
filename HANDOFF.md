# Local Agent Handoff: Zoology RMT/MQAR

Use this as compact context for a fresh agent. This file is local-only and is
listed in `.git/info/exclude`; do not commit it. Update it only when the user
explicitly asks.

## Repository

- Path: `/home/admin/cbp/nlp-cl/zoology-mqar-fmrmt`
- Fork/workspace: HazyResearch/zoology
- Goal: explore Base RMT and Fast Memory RMT on MQAR-style associative recall,
  then build toward continual-learning claims.
- User shell: fish in VS Code terminal.
- Python env: `.venv`; use `.venv/bin/python` when running from tools.
- Do not run long experiments unless the user asks. Cheap tests/import checks
  are fine.

## Implemented Architecture/Training Pieces

- `zoology/mixers/rmt.py`
  - `BaseRMTMixer`
  - `FastMemoryRMTMixer`
  - Current RMT layout is read/write memory: `[M_read, segment_tokens, M_write]`.
  - Fast memory has manual update hooks used by the trainer.
- `zoology/train.py`
  - `train_continual(config)` trains one model through stages with persistent
    weights and optimizer state.
  - Fast-memory params are excluded from AdamW and updated through the mixer hook.
  - Logs RMT diagnostics such as memory norm, update norm, and grad norm.
  - Formal CL metrics were added:
    - `continual/plasticity`
    - `continual/current_stage_accuracy`
    - `continual/stage_{i}/learning_accuracy`
    - `continual/stage_{i}/bwt`
    - `continual/avg_bwt`
    - `continual/stage_{i}/forgetting_from_learning`
    - `continual/avg_forgetting_from_learning`
    - optional FWT metrics with `evaluate_future_stages=True`
- `zoology/config.py`
  - `ContinualDataConfig`
  - `ContinualTrainConfig`
  - `evaluate_future_stages: bool = False`
- `zoology/logger.py`
  - Comet support exists; local JSON capture runners have also been used.

## Data/Experiment Files

- `zoology/data/class_incremental_ar.py`
  - Main class-incremental associative retrieval task.
  - Fixed global vocab; each stage introduces disjoint key/value classes.
  - Current formal CL configs use stage-local `eval_mode="current"` test splits,
    so plasticity and forgetting are not conflated.
- `zoology/data/incremental_mqar.py`
  - Vocab-growth MQAR; currently not a clean ranking benchmark.
- `zoology/data/interference_ar.py`
  - Repeated-key interference task with `target_policy="latest"` or `"old"`.
- `zoology/experiments/rmt_mqar/class_incremental_ar.py`
  - Clean 5-stage formal CL reference run.
- `zoology/experiments/rmt_mqar/class_incremental_ar_horizon.py`
  - Horizon sweep over 5, 10, 20 stages.
- `zoology/experiments/rmt_mqar/class_incremental_ar_interference.py`
  - No-conflict, repeated-key latest-value, repeated-key old-value variants.
- Docs:
  - `README.md`
  - `ARCHITECTURES.md`
  - `CONTINUAL_MQAR.md`
  - `RESULTS.md`

## Important Caveat About Results

Results before 2026-05-13 used cumulative-prefix class-incremental evaluation.
Current formal CL runs use stage-local test splits. Do not directly compare
old cumulative-prefix numbers to the formal CL metrics.

## Key Result Artifacts

- `results/class_incremental_ar_formal_cl_20260513_022211.json`
  - Clean 5-stage formal CL reference.
  - FMRMT stable: seen avg `0.8501`, plasticity `1.0`, avg BWT `-0.1874`.
  - Base RMT: seen avg `0.8081`, plasticity `1.0`, avg BWT `-0.2399`.
  - Transformer: seen avg `0.1321`, plasticity `0.4832`, avg BWT `-0.9504`.
  - Interpretation: Base RMT and FMRMT both learn new stages perfectly; FMRMT
    retains better. Transformer forgets badly and loses plasticity.
- `results/class_incremental_ar_horizon_formal_cl_20260513_054644.json`
  - Horizon sweep over 5/10/20 stages, 3 seeds, lower budget than reference
    (`2048` train examples/stage, `8` epochs/stage).
  - Pattern from live summary:
    - FMRMT retains much better, with lower forgetting/BWT loss.
    - FMRMT shows late-stage plasticity failure at 10/20 stages.
    - Base RMT preserves plasticity better at 10 stages but forgets more.
    - At 20 stages both Base RMT and FMRMT can fail final-stage plasticity,
      while FMRMT still retains older stages better.
- `results/class_incremental_ar_interference_formal_cl_20260513_055618.json`
  - Repeated-key interference sweep.
  - No conflict: FMRMT has stronger retention than Base RMT.
  - Repeated key, latest value: hard overwrite case.
    - Base RMT often learns updates better but forgets more.
    - FMRMT retains better but is less plastic.
  - Repeated key, old value: easy preservation case; FMRMT retains best.
- Historical:
  - `results/best_arch_class_incremental_ar_20260511_200731.json`
  - `results/fmrmt_tuning_class_incremental_ar_20260511_193201.json`
  - `results/class_incremental_ar_end_to_end_20260511_173007.json`
  - `results/vocab_growth_diagnostics_20260511_203603.json`

## Current Best Model Setting

Stable FMRMT candidate:

```text
n_mem = 8
rmt_fast_lr = 0.005
rmt_slow_update_freq = 4
reset_memory_each_batch = False
reset_memory_each_epoch = True
rmt_clip_memory_grad = 1.0
segment_len = 64
```

This setting is strong for retention, but the latest horizon/interference runs
show a plasticity tradeoff under longer streams and latest-value overwrite.

## Next Suggested Work

1. Update `RESULTS.md` with the horizon and interference summaries from:
   - `results/class_incremental_ar_horizon_formal_cl_20260513_054644.json`
   - `results/class_incremental_ar_interference_formal_cl_20260513_055618.json`
2. Tune FMRMT for plasticity on the hard cases:
   - 10-stage horizon.
   - repeated-key latest-value interference.
   - Sweep `rmt_fast_lr`, `rmt_slow_update_freq`, and reset policy.
3. Keep claims precise:
   - Current FMRMT improves retention/BWT.
   - Current FMRMT can sacrifice plasticity in harder streams.
   - Base RMT is more plastic in some harder settings but forgets more.
4. After tuning, rerun the best configs with 5 seeds.

## Useful Commands

Run reference formal CL:

```fish
set -x CUDA_VISIBLE_DEVICES 2
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar.py --gpus 1
```

Run horizon:

```fish
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar_horizon.py --gpus 1
```

Run interference:

```fish
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar_interference.py --gpus 1
```

Cheap validation:

```bash
.venv/bin/python -m pytest tests/test_class_incremental_ar.py tests/test_interference_ar.py tests/test_continual_mqar.py tests/test_rmt_imports.py -q
```

Last known cheap validation before this note:

```text
19 passed
```
