#!/bin/bash
# Train RMG on MotionMillion (~458k released rpr272 clips, 30fps). Same model + recipe as the HumanML3D
# run, lazy DataLoader over the big set. Run mm_setup.sh + mm_prep.sh first. Run from the repo root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config.sh" 2>/dev/null || { echo "copy config.example.sh -> config.sh and set your paths"; exit 1; }
python src/rmg_train_mm.py \
  --index cache/mm_train_index.npz --emb cache/mm_train_emb.npy \
  --steps ${STEPS:-300000} --batch ${BATCH:-64} --accum ${ACCUM:-4} --workers ${WORKERS:-12} \
  --dim 384 --layers 6 --heads 8 --ff_mult 8 \
  --lr 1e-4 --warmup_frac 0.08 --clip 0.5 --ema_decay 0.9999 --p_drop 0.1 \
  --out runs/rmg_mm
