# Valid Incremental AR Experimental Setup

This note separates the benchmark design, the comparisons it supports, and the
claims that are already backed by current runs.

## Scope

RMT and FastMem RMT should be described as model architectures with recurrent
memory / online memory adaptation, not as continual-learning algorithms in the
same category as EWC, SI, or replay. The paper comparison should therefore use
three groups:

1. Core online architectures: Transformer/MHA, Base RMT, FastMem RMT, and no-fast
   FastMem controls.
2. Standard CL methods on Transformer/MHA: online EWC, Synaptic Intelligence,
   reservoir replay, and stored-buffer replay.
3. Hybrid architecture + replay controls: Base RMT + stored replay and FastMem
   RMT + stored replay.

## Primary Benchmark

Use `formal20_permuted` class-incremental AR as the main benchmark.

- 20 sequential stages.
- Fixed per-stage key-value table.
- `value_mapping="permuted"` with `association_table_seed=20260522`.
- Stage-local train and stage-local test splits.
- Train only on the current stage unless the run is explicitly a replay
  baseline.
- Evaluate all seen stages after every stage.
- Seeds: `123,456,789`.
- Scheduler: `stage_onecycle`.
- Slow updates: `slow_update_mode="accumulate"`.
- Metrics:
  - `continual/plasticity`: current-stage final accuracy after learning.
  - `continual/avg_learning_accuracy`: diagonal of the stage-end accuracy
    matrix.
  - `continual/seen_avg_accuracy`: final average over all seen stage-local
    tests.
  - `continual/avg_bwt`: final accuracy minus learning-time accuracy on old
    stages.
  - `continual/avg_forgetting_from_learning`: learning-time accuracy minus final
    accuracy, clipped at zero.
  - Stage-end accuracy matrix and current-stage epoch curves.

This setup is valid for measuring retention and BWT in a sequential AR stream.
It is not a pure plasticity-loss benchmark, because most RMT-family variants
learn each current stage to perfect top-1 accuracy.

## Required Validity Additions

Top-1 accuracy alone is too coarse. Some runs show epoch curves like
`0.0 -> 0.0 -> 1.0` while loss decreases smoothly, meaning the model crosses a
top-1 decision boundary before its representation is necessarily well
consolidated.

Before the final paper runs, add and report at least:

- current-stage test loss / NLL at learning time;
- correct-token probability at masked query positions;
- logit margin: correct value logit minus best wrong value logit;
- optionally per-key accuracy.

These metrics distinguish "learned by top-1" from "learned robustly." The
existing accuracy matrices are enough for BWT/forgetting claims, but not enough
for a strong consolidation claim.

## Main Core Result

Source:
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033410_720075.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033413_865324.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_033421_414890.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_109006.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_124162.json`
`results/class_incremental_ar_permuted_formal_stage_onecycle_accumulate_formal20_permuted_20260524_040149_814041.json`

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0419 +/- 0.0131 | 0.8577 +/- 0.0544 | 0.7928 +/- 0.2677 | -0.8588 +/- 0.0517 | 0.8158 +/- 0.0491 |
| Base RMT n_mem=16 | 0.8121 +/- 0.0120 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.1978 +/- 0.0126 | 0.1879 +/- 0.0120 |
| FastMem RMT lr=0.005 slow_freq=2 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 |
| FastMem RMT fast_lr=0 slow_freq=2 | 0.9774 +/- 0.0073 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0238 +/- 0.0076 | 0.0226 +/- 0.0073 |

Supported claim:

- RMT-style memory plus the stage-local schedule gives much stronger online
  continual retention than a plain Transformer/MHA.
- FastMem RMT is a strong online architecture in this setup.

Unsupported claim:

- These runs do not prove that the fast-memory gradient update is the causal
  source of the gain, because the `fast_lr=0` control is almost identical to
  `fast_lr=0.005`.

## Regularization CL Baselines

Sources:
`results/class_incremental_ar_permuted_online_ewc_formal20_permuted_20260524_*.json`
`results/class_incremental_ar_permuted_si_formal20_permuted_20260524_*.json`

| Method | Best Setting | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Online EWC | lambda=100000 | 0.2705 +/- 0.1772 | 0.9916 +/- 0.0119 | 1.0000 +/- 0.0000 | -0.7590 +/- 0.1990 | 0.7211 +/- 0.1891 |
| SI | lambda=300 | 0.8594 +/- 0.1157 | 0.9230 +/- 0.0564 | 0.7306 +/- 0.2985 | -0.0670 +/- 0.0909 | 0.0865 +/- 0.1162 |
| FastMem RMT no replay | lr=0.005 slow_freq=2 | 0.9788 +/- 0.0154 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0224 +/- 0.0162 | 0.0212 +/- 0.0154 |

Supported claim:

- SI is a real comparator and much stronger than EWC, but its best retention
  setting has a plasticity cost. FastMem RMT keeps both high retention and
  perfect current-stage learning on this benchmark.

## Replay CL Baselines

Stored-buffer replay is the cleanest replay comparison because old examples are
fixed stored examples rather than regenerated examples.

Source:
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_stored_replay*_formal20_permuted_20260524_*.json`

| Model | Stored Replay/Old Stage | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 1 | 0.0759 +/- 0.0375 | 0.8224 +/- 0.1264 | 0.9709 +/- 0.0411 | -0.7858 +/- 0.0940 | 0.7465 +/- 0.0893 |
| Transformer/MHA | 8 | 0.9312 +/- 0.0926 | 0.8941 +/- 0.0572 | 1.0000 +/- 0.0000 | 0.0391 +/- 0.1080 | 0.0415 +/- 0.0564 |
| Transformer/MHA | 16 | 1.0000 +/- 0.0000 | 0.9722 +/- 0.0393 | 1.0000 +/- 0.0000 | 0.0292 +/- 0.0413 | 0.0000 +/- 0.0000 |
| Base RMT n_mem=16 | 1 | 0.9502 +/- 0.0091 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0524 +/- 0.0095 | 0.0498 +/- 0.0091 |
| Base RMT n_mem=16 | 2 | 0.9712 +/- 0.0108 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0303 +/- 0.0114 | 0.0288 +/- 0.0108 |
| FastMem RMT | 1 | 0.9874 +/- 0.0128 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0133 +/- 0.0135 | 0.0126 +/- 0.0128 |
| FastMem RMT | 2 | 0.9983 +/- 0.0012 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0017 +/- 0.0013 | 0.0017 +/- 0.0012 |

Supported claim:

- Replay can solve the task if the buffer is large enough.
- RMT-style architectures are much more replay-efficient than Transformer/MHA.
- The strongest current FastMem RMT result is the tiny-buffer regime:
  FastMem RMT with one stored example per old stage outperforms Base RMT with
  one or two stored examples per old stage, and it reaches near-perfect
  retention with two stored examples per old stage.

## Standard Reservoir Replay

Source:
`results/class_incremental_ar_permuted_replay_stage_onecycle_accumulate_fixed_total_reservoir_replay*_buffer*_formal20_permuted_20260524_*.json`

| Model | Reservoir Capacity | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA + reservoir | 19 | 0.0535 +/- 0.0216 | 0.7833 +/- 0.1623 | 0.7870 +/- 0.3012 | -0.7682 +/- 0.1753 | 0.7298 +/- 0.1666 |
| Transformer/MHA + reservoir | 38 | 0.1470 +/- 0.0845 | 0.7431 +/- 0.2070 | 1.0000 +/- 0.0000 | -0.6274 +/- 0.1388 | 0.5961 +/- 0.1318 |
| Transformer/MHA + reservoir | 76 | 0.6646 +/- 0.2260 | 0.8131 +/- 0.1327 | 1.0000 +/- 0.0000 | -0.1563 +/- 0.0984 | 0.1671 +/- 0.0873 |
| Transformer/MHA + reservoir | 152 | 0.8388 +/- 0.1984 | 0.9503 +/- 0.0550 | 1.0000 +/- 0.0000 | -0.1173 +/- 0.1519 | 0.1162 +/- 0.1408 |
| Transformer/MHA + reservoir | 304 | 0.9999 +/- 0.0001 | 0.9907 +/- 0.0131 | 1.0000 +/- 0.0000 | 0.0097 +/- 0.0137 | 0.0001 +/- 0.0001 |

Supported claim:

- Standard reservoir replay is a strong baseline at moderate capacity, but tiny
  reservoirs are much weaker than core FastMem RMT and Base RMT.

## Stress / Weakness Tests

The main stress branch to keep is:

- `formal20_permuted`
- `input_seq_len=512`
- `segment_len=64`
- `vocab_size=4096`
- `associations_per_stage=32`
- `num_query_associations=16`
- 512 train examples per stage
- 512 test examples per stage
- 8 epochs per stage

Current one-seed result:

| Model | Seen Avg Accuracy | Avg Learning Accuracy | Final Plasticity | Avg BWT | Forgetting From Learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer/MHA | 0.0015 | 0.1873 | 0.0295 | -0.1956 | 0.1858 |
| Base RMT n_mem=16 | 0.9449 | 1.0000 | 1.0000 | -0.0580 | 0.0551 |
| Base RMT n_mem=8 | 0.8893 | 1.0000 | 1.0000 | -0.1165 | 0.1107 |
| FastMem RMT fast_lr=0 slow_freq=2 | 0.8987 | 1.0000 | 1.0000 | -0.1066 | 0.1013 |
| FastMem RMT lr=0.005 slow_freq=2 | 0.9191 | 1.0000 | 1.0000 | -0.0851 | 0.0809 |

Supported claim if confirmed over three seeds:

- This shape can expose a small benefit of the nonzero fast-memory update over
  the no-fast control while still showing that high-capacity Base RMT remains a
  strong competitor.

Current caveat:

- This is one seed. It should not be used as a final paper claim until rerun
  over seeds `123,456,789` with the robustness metrics listed above.

Do not escalate the `assoc64` variant as currently configured. It makes Base
RMT almost solve the task and does not produce a cleaner FastMem comparison.

## Final Recommended Paper Tables

1. Core formal20 permuted AR:
   Transformer/MHA, Base RMT n_mem=16, FastMem RMT lr=0.005 slow2, FastMem
   RMT fast_lr=0 slow2, SI best, EWC best.

2. Replay-efficiency formal20 permuted AR:
   Transformer/MHA stored replay `{1, 8, 16}`, Base RMT stored replay `{1, 2}`,
   FastMem RMT stored replay `{1, 2}`. Include reservoir replay `{19, 76, 152,
   304}` as the standard ER reference.

3. Mechanism / ablation table:
   Fast LR sweep at slow2, no-fast control, segment_len=128 control, and the
   seq512 stress diagnostic after three seeds.

4. Heatmaps:
   Use stage-end accuracy matrices and forgetting-from-learning matrices.
   Label them as retention/forgetting, not plasticity.

## Final Paper Claim That Is Currently Defensible

FastMem RMT is not yet proven to beat all alternatives because of its local
fast-memory gradient update alone. The defensible claim is:

> In incremental associative retrieval without replay, RMT-style memory
> architectures retain stage-local associations far better than a plain
> Transformer and stronger than online EWC or SI while preserving current-stage
> learning. With tiny stored replay, FastMem RMT is more replay-efficient than
> Base RMT and much more replay-efficient than Transformer/MHA. Larger replay
> buffers can solve the task, so the advantage is specifically in the core
> online and tiny-buffer regimes.

The strongest unresolved mechanism question is whether the fast-memory update
improves consolidation beyond the slow-update/memory-augmented architecture. The
current formal20 core data says "not clearly"; the tiny-buffer and seq512
stress diagnostics are the best places to test it.
