#!/bin/bash
# Precompute the MotionMillion clip index + Qwen caption embeddings (run once before training).
# Streams texts.tar.gz, encodes captions with Qwen3-Embedding-0.6B, writes cache/mm_<split>_{index.npz,emb.npy}.
# Run from the repo root. Set paths (incl. HF_HOME) in config.sh.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config.sh" 2>/dev/null || { echo "copy config.example.sh -> config.sh and set your paths"; exit 1; }
mkdir -p cache
python src/mm_prep.py --split train --caps 4 --out cache/mm     # ~458k clips, a few captions each
python src/mm_prep.py --split val   --caps 1 --out cache/mm
