# RMT MQAR Experiments

This folder adds starting MQAR experiment scaffolding for comparing existing
Transformer/MHA baselines with two RMT variants:

- `BaseRMTMixer` in `zoology/mixers/rmt.py`
- `FastMemoryRMTMixer` in `zoology/mixers/rmt.py`

`BaseRMTMixer` uses trainable memory tokens and processes sequences in segments.
Each segment is run as `[memory tokens, segment tokens]`; updated memory is
carried to the next segment, and only non-memory outputs are returned.

`FastMemoryRMTMixer` shares the same segmented path and adds a fast-memory tensor
updated by the trainer after `loss.backward()`. The trainer logs memory norm,
memory update norm, and memory gradient norm when present. WandB is optional; the
configs here leave `LoggerConfig` empty by default.

Local experiment summaries are tracked in `RESULTS.md`.


## Files

- `smoke.py`: tiny launch check with Transformer/MHA, Base RMT, and Fast Memory RMT.
- `capacity_sweep.py`: compares the same models while increasing `num_kv_pairs`.
- `context_sweep.py`: compares the same models while increasing `input_seq_len`.
- `interference_sweep.py`: static repeated-key MQAR check using
  `ForgettingMQARConfig`.
- `continual_vocab.py`: sequential continual MQAR where one model is trained
  through disjoint key/value vocab stages and evaluated on all seen stages after
  each stage.
- `class_incremental_ar.py`: class-incremental associative retrieval where each
  stage introduces new fixed key/value association classes, trains only on the
  current stage, and evaluates all seen stages with stage-local test splits.
- `class_incremental_ar_horizon.py`: harder class-incremental AR horizon sweep
  over more stages, with future-stage/pre-learning metrics enabled.
- `class_incremental_ar_interference.py`: continual repeated-key interference
  variants with no-conflict, latest-value, and old-value targets.
- `_common.py`: shared model and train-config factories.
- `ARCHITECTURES.md`: explicit architecture descriptions and diagrams for the
  Transformer/MHA baseline, Base RMT, and Fast Memory RMT.
- `CONTINUAL_MQAR.md`: explanation of the sequential vocab-growth continual
  MQAR setup, metrics, defaults, and interpretation.
- `RESULTS.md`: curated summary of the most important local runs, with links to
  the raw JSON artifacts in `results/`.

## Run Later

Environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

For GPU runs, replace the CPU PyTorch line with the CUDA wheel command that
matches the machine's NVIDIA driver.

Optional extras for parallel sweeps:

```bash
pip install -e ".[extras]"
```

Optional CUDA-extension mixers, not needed for RMT/MQAR smoke runs:

```bash
pip install -e ".[cuda_mixers]"
```

## Comet Logging

The trainer logs to Comet when `COMET_PROJECT_NAME` or `COMET_API_KEY` is set.
Each config in a sweep becomes one Comet experiment named by `run_id`, for
example `attention-kv8-seq64` or `fast_memory_rmt-kv64-seq512`.

In fish:

```fish
set -x COMET_API_KEY "<your-api-key>"
set -x COMET_WORKSPACE "<your-workspace>"
set -x COMET_PROJECT_NAME "zoology-rmt-mqar"
```

For offline logging:

```fish
set -x COMET_OFFLINE 1
```

The logged metrics include:

- `train/loss`
- `valid/loss`
- `valid/accuracy`
- continual metrics for `continual_vocab.py` and `class_incremental_ar.py`, including
  `continual/seen_avg_accuracy`, `continual/avg_forgetting`,
  `continual/plasticity`, `continual/avg_bwt`,
  `continual/avg_fwt_from_random`, and per-stage
  `continual/stage_{i}/accuracy`
- sliced validation accuracy, such as `valid/num_kv_pairs/accuracy-64`
- RMT diagnostics, such as `train/rmt_memory_norm`,
  `train/rmt_memory_grad_norm`, and `train/rmt_memory_update_norm`

Smoke check:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/smoke.py
```

Capacity sweep:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/capacity_sweep.py
```

Context sweep:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/context_sweep.py
```

Interference sweep:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/interference_sweep.py
```

Continual vocab-growth MQAR:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/continual_vocab.py --gpus 1
```

Class-incremental associative retrieval:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar.py --gpus 1
```

Class-incremental horizon sweep:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar_horizon.py --gpus 1
```

Class-incremental interference sweep:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/class_incremental_ar_interference.py --gpus 1
```

Multi-GPU parallel launch, if supported by current Zoology:

```bash
python -m zoology.launch zoology/experiments/rmt_mqar/capacity_sweep.py -p
python -m zoology.launch zoology/experiments/rmt_mqar/context_sweep.py -p
```

The default config sizes are intentionally small. Scale `num_examples`,
`max_epochs`, `CAPACITY_POINTS`, and `CONTEXT_LENGTHS` before running full sweeps.
