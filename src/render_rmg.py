"""
Quick visual sanity for an RMG checkpoint: generate conditioned motion for distinct captions (CFG) and
render skeleton GIFs with their prompts. The fastest "are we on the right path?" check — a correct
S^3/RFM pipeline yields recognizable, text-appropriate human motion even before convergence.

    python render_rmg.py --ckpt runs/rmg_base/model.pth --guidance 6.5 --out report
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ.get("HML3D_REPO", os.path.expanduser("~/hml3d_repo")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.float = float
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from eval_hml import positions, _skel
import s3
import qwen_text
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow

PARENTS = _skel.parents()
CAPTIONS = [
    "a person walks forward",
    "a person sits down",
    "a person raises both arms above the head",
    "a person kicks with the right leg",
    "a person jumps",
    "a person turns around",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rmg_base/model.pth")
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--L", type=int, default=120)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    c = torch.load(a.ckpt, map_location=dev)
    model = RMGTransformer(**c["config"]).to(dev)
    key = "state_dict" if a.weights == "raw" else "ema_state_dict"
    model.load_state_dict(c.get(key, c["state_dict"])); model.eval()
    flow = RMGFlow(sigma_trans=c.get("sigma_trans", 1.0), sigma_rot=c.get("sigma_rot", 1.0))
    step = c.get("step", "?")

    torch.manual_seed(a.seed)
    cond = qwen_text.encode(CAPTIONS, device=dev).to(dev)
    tr, q = flow.sample(model, len(CAPTIONS), a.L, text=cond, guidance=a.guidance, n_steps=a.steps, device=dev)
    R = s3.quat_to_matrix(q.cpu())
    P = torch.stack([positions(tr[i].cpu(), R[i]) for i in range(len(CAPTIONS))]).numpy()

    n, cols = len(CAPTIONS), 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    axes = [fig.add_subplot(rows, cols, i + 1, projection="3d") for i in range(n)]
    lims = [[(P[i].reshape(-1, 3)[:, d].min() - .2, P[i].reshape(-1, 3)[:, d].max() + .2) for d in range(3)]
            for i in range(n)]

    def draw(fr):
        for i, ax in enumerate(axes):
            ax.cla()
            p = P[i, fr]
            for j, par in enumerate(PARENTS):
                if par >= 0:
                    ax.plot([p[par, 0], p[j, 0]], [p[par, 2], p[j, 2]], [p[par, 1], p[j, 1]], c="#c0392b", lw=2.2)
            ax.set_xlim(*lims[i][0]); ax.set_ylim(*lims[i][2]); ax.set_zlim(*lims[i][1])
            ax.set_box_aspect((1, 1, 1.6)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f'"{CAPTIONS[i]}"', fontsize=9)
        fig.suptitle(f"RMG-base (step {step}, guidance {a.guidance})   frame {fr+1}/{a.L}", fontsize=13)
        return []

    path = f"{a.out}/gallery_rmg.gif"
    FuncAnimation(fig, draw, frames=a.L, interval=60).save(path, writer=PillowWriter(fps=16))
    plt.close(fig)
    print("wrote", path, "| step", step)


if __name__ == "__main__":
    main()
