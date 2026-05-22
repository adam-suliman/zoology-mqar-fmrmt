# Main Incremental AR Sample Flow

This note describes how samples are constructed, batched, received by the model,
and evaluated in the main class-incremental associative-retrieval experiment:

`zoology/experiments/rmt_mqar/class_incremental_ar.py`

## Main Settings

| Setting | Value |
| --- | ---: |
| Stages | 5 |
| Vocabulary size | 1024 |
| Associations per stage | 16 |
| Queried associations per sample | 8 |
| Input sequence length | 128 |
| Train examples per stage | 4096 |
| Test examples per stage | 1024 |
| Batch size | 64 |
| Epochs per stage | 16 |
| RMT segment length | 64 |

Each stage is trained and tested with `eval_mode="current"` and
`distractor_mode="current"`. That means stage `s` samples only from stage `s`'s
own key/value block. Forgetting is measured by revisiting earlier stage-local
test sets after later stages have been trained.

## Stage Association Table

The vocabulary is split into a key half and a value half:

```text
key token IDs   : 1..511        token 0 is reserved during generation
value token IDs : 512..1023
```

Each stage owns a disjoint block of 16 key/value associations. For stage `s`:

```text
key IDs   = 1 + 16*s      ... 1 + 16*s + 15
value IDs = 512 + 16*s    ... 512 + 16*s + 15
```

The mapping is offset-aligned inside the block:

```text
stage 0:
  key 1  -> value 512
  key 2  -> value 513
  ...
  key 16 -> value 527

stage 1:
  key 17 -> value 528
  key 18 -> value 529
  ...
  key 32 -> value 543
```

So `associations_per_stage = 16` means 16 possible key/value mappings for that
stage. It does not mean that all 16 mappings appear in every sequence.


### Random-Permuted Mapping Variant

The default main formal experiment uses `value_mapping="aligned"`, shown above.
The new control uses `value_mapping="permuted"`. It keeps the same disjoint
key and value blocks per stage, but randomly permutes the value order inside
that stage block:

```text
stage s keys   : k0, k1, ..., k15
stage s values : perm(v0, v1, ..., v15)
```

The permutation is fixed by `association_table_seed + stage_idx`, not by the
data sampling seed. That gives the control we want:

```text
same association_table_seed -> same key/value table across train and test
different data seed         -> different sampled sequences/query positions
```

This removes the arithmetic shortcut where `value_id - 512` equals
`key_id - 1`, while preserving the class-incremental structure: each stage still
owns 16 fixed associations and each sample still draws a subset from the current
stage table.

## Single Sample Construction

For one sample in stage `s`:

1. Pick 8 distinct associations from the 16 associations available in stage `s`.
2. Put those 8 key/value pairs at the beginning of the sequence.
3. Choose 8 query positions later in the sequence.
4. Put the queried keys at those query positions.
5. Set the target label at each query position to the corresponding value.
6. Fill all unused zero positions with random current-stage key/value distractor
   tokens.

The first part of each sample is always the explicit key/value context:

```text
position:  0   1   2   3        14  15
token:    kA  vA  kB  vB  ...   kH  vH
```

Because there are 8 queried associations and each pair is two tokens:

```text
context tokens = 8 pairs * 2 tokens = 16 tokens
```

The remaining 112 input positions are the query/distractor region:

```text
positions 16..127
```

At 8 selected positions in that region, the input token is a queried key and
the label is the matching value:

```text
input token at query position : kA
target label at same position : vA
```

All non-query positions have label `-100`, so they are ignored by
cross-entropy loss and accuracy computation.

An abstract sample looks like:

```text
inputs:
  kA vA  kB vB  kC vC  kD vD  kE vE  kF vF  kG vG  kH vH
  d  d   kC d   d  d   kA d   d  kH  d  d   ...    kF d

labels:
  -  -   -  -   -  -   -  -   -  -   -  -   -  -   -  -
  -  -   vC -   -  -   vA -   -  vH  -  -   ...    vF -
```

Here `d` is a random distractor token from the current stage's key/value block,
and `-` means the actual label is `-100`.

## Dataset and Batches

For each stage there is a separate train split and test split:

```text
stage 0 train: 4096 samples
stage 0 test : 1024 samples
stage 1 train: 4096 samples
stage 1 test : 1024 samples
...
stage 4 train: 4096 samples
stage 4 test : 1024 samples
```

With batch size 64:

```text
train batches per epoch = 4096 / 64 = 64
test batches per stage  = 1024 / 64 = 16
```

With 16 epochs per stage:

```text
train batches per stage = 64 * 16 = 1024
```

The dataloader yields already-batched objects:

```text
inputs : [batch_size, 128]
labels : [batch_size, 128]
slices : metadata dicts copied across the batch
```

The important slice metadata includes:

```text
stage_idx
eval_mode
associations_per_stage
num_query_associations
input_seq_len
```

## Continual Training Order

The experiment uses one model trained sequentially through the five stage
dataloaders:

```text
train stage 0 for 16 epochs
evaluate stages 0

train stage 1 for 16 epochs
evaluate stages 0, 1

train stage 2 for 16 epochs
evaluate stages 0, 1, 2

train stage 3 for 16 epochs
evaluate stages 0, 1, 2, 3

train stage 4 for 16 epochs
evaluate stages 0, 1, 2, 3, 4
```

This produces an accuracy matrix:

```text
A[t, i] = accuracy on test stage i after training through stage t
```

The diagonal is stage learning accuracy:

```text
A[i, i] = accuracy on stage i immediately after learning stage i
```

The final row is final retention:

```text
A[4, i] = final accuracy on stage i after all stages are trained
```

Average BWT is computed from old-stage final-row values versus the diagonal:

```text
avg_bwt = mean_i<4 (A[4, i] - A[i, i])
```

## Loss and Accuracy

The model predicts a vocabulary distribution at every sequence position, but
loss is only applied at query positions.

Implementation detail:

```text
labels == -100  -> ignored
labels != -100  -> query target value
```

So the task is not to reconstruct the full sequence. The task is to output the
correct value token when the input position contains a queried key.

## Transformer Processing

The Transformer/MHA baseline receives the full sequence at once:

```text
[batch, 128] token IDs
-> token/position embeddings
-> full 128-token self-attention stack
-> logits over vocabulary at each position
-> loss only at query positions
```

It has no recurrent segmentation and no explicit memory tokens.

## Base RMT Processing

Base RMT receives the same 128-token input but processes it in two 64-token
segments:

```text
segment 0: positions 0..63
segment 1: positions 64..127
```

For each segment, the mixer builds:

```text
[read memory, segment tokens, write memory]
```

The write-memory output from segment 0 becomes the read-memory input for
segment 1:

```text
segment 0 -> write memory -> segment 1
```

In the main experiment Base RMT uses:

```text
n_mem = 4
segment_len = 64
memory_layout = read_write
detach_memory_between_segments = True
```

The hidden segment memory is carried only inside the current sample's forward
pass. It is not the hidden state of the previous sample. Across samples and
stages, Base RMT transfers information only through learned parameters:

```text
model weights
token embeddings
learned memory token parameters
```

## FMRMT Processing

FMRMT uses the same segmented RMT structure, but its memory tokens are split
into slow initial memory and explicit fast memory.

In the main stable-FMRMT experiment:

```text
n_mem = 8
segment_len = 64
rmt_fast_lr = 0.005
rmt_slow_update_freq = 4
reset_memory_each_batch = False
reset_memory_each_epoch = True
rmt_clip_memory_grad = 1.0
detach_memory_between_segments = False
```

During training:

1. The forward pass initializes segment memory from `fast_memory_tokens`.
2. The sequence is processed in two 64-token segments.
3. Backprop computes gradients for model weights and fast memory.
4. `update_fast_memory()` updates fast memory every batch.
5. Slow model weights update every `rmt_slow_update_freq` batches.
6. Because `reset_memory_each_epoch=True`, fast memory resets to the learned
   initial memory at every epoch boundary.

So FMRMT can carry fast-memory state across batches within the same epoch when
`reset_memory_each_batch=False`, but it does not carry that fast-memory state
across epochs in the main stable config.

During evaluation, FMRMT initializes memory from learned `initial_memory_tokens`,
not from the current training fast-memory state. This matters for interpretation:
reported test accuracy measures the learned model state, not a hidden state that
has been carried through the whole training stream.

## What Transfers Across Stages

After finishing stage `s`, the model proceeds to stage `s+1` with the same
learned parameters:

```text
embeddings
attention/RMT block weights
MLP/state-mixer weights
learned RMT initial memory parameters
```

For Base RMT, there is no persistent hidden memory state transferred from the
last sample of stage `s` to the first sample of stage `s+1`.

For stable FMRMT, fast memory is reset at epoch boundaries, and evaluation uses
initial memory. The main cross-stage transfer is therefore still through learned
parameters, with fast memory acting as a training-time update mechanism rather
than a permanent hidden state passed through all stages.

