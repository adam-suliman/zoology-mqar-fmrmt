#!/usr/bin/env bash
set -euo pipefail

# Smoke
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \
  --num-stages 3 \
  --seeds 123 \
  --examples-per-seen-stage 256 \
  --test-examples-per-stage 128 \
  --epochs-per-stage 2 \
  --output-prefix transformer_cumulative_ar_plasticity_smoke

# Full 20-stage run
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \
  --num-stages 20 \
  --seeds 123,456,789 \
  --examples-per-seen-stage 1024 \
  --test-examples-per-stage 512 \
  --epochs-per-stage 8 \
  --output-prefix transformer_cumulative_ar_plasticity_20stage

# Escalation if no loss appears in the 20-stage run
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} .venv/bin/python -m zoology.experiments.rmt_mqar.class_incremental_ar_transformer_cumulative_plasticity \
  --num-stages 50 \
  --seeds 123 \
  --examples-per-seen-stage 1024 \
  --test-examples-per-stage 512 \
  --epochs-per-stage 8 \
  --vocab-size 2048 \
  --output-prefix transformer_cumulative_ar_plasticity_50stage_seed123
