#!/bin/bash
# RMG-base reproduction on HumanML3D (effective batch 256 = 32 x 8 accum, 150k steps).
# Run from the repo root. Activate your env and `source config.sh` first (copy config.example.sh).
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; source "$ROOT/config.sh" 2>/dev/null || true
python src/rmg_train.py --steps 150000 --batch 32 --accum 8 --dim 384 --layers 6 --heads 8 --ff_mult 8 \
  --lr 1e-4 --warmup_frac 0.08 --clip 0.5 --ema_decay 0.9999 --p_drop 0.1 \
  --cache cache_rmg_train.pt --out runs/rmg_base
