# Copy to config.sh and fill in the paths for your machine, then `source config.sh`.
# config.sh is gitignored on purpose -- keep real paths/hostnames out of the repo.
#
#   cp config.example.sh config.sh && $EDITOR config.sh

# --- datasets ---
export HML_DIR="${HML_DIR:-/path/to/HumanML3D}"            # humanml3d root (new_joint_vecs, texts, splits)
export MM_ROOT="${MM_ROOT:-/path/to/MotionMillion}"        # motionmillion root (folder*/, texts.tar.gz, mean_std, split.tar.gz)
export MM_META="${MM_META:-./cache/mm_meta}"               # where mm_setup.sh extracts the split lists

# --- models / caches ---
export SMPLX_PATH="${SMPLX_PATH:-/path/to/smplx/models}"   # smpl-x model files (for the body mesh)
export HF_HOME="${HF_HOME:-./cache/hf}"                    # huggingface cache (qwen encoder lands here)

# --- eval (humanml3d official evaluators, see scripts/setup.sh) ---
export T2M_EVAL="${T2M_EVAL:-./data/t2m_eval}"
export T2M_REPO="${T2M_REPO:-./external/text-to-motion}"
export HML3D_REPO="${HML3D_REPO:-./external/HumanML3D}"

# --- checkpoint ---
export RMG_CKPT="${RMG_CKPT:-runs/rmg_base/model.pth}"
