#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

DATA_DIR="${DATA_DIR:-/path/to/the/dataset}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" train.py \
  --data_dir "$DATA_DIR" \
  --split_train train \
  --batch_size 24 \
  --ncpus 20 \
  --lr 1e-4 \
  --lr_decay_mode step \
  --lr_decay_step 10000 \
  --lr_decay_gamma 0.5 \
  --max_steps 50000 \
  --save_interval 10000 \
  --use_fp16 false \
  --max_norm 1.0 \
  --temperatures 1.0,1.5 \
  --lambda_l1 0 \
  --checkpointdir checkpoints/Custom/RDDM-Fine
