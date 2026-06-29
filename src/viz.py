"""
Visualization for full-body motion: animated skeletons (the payoff) + a smoothness diagnostic.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import skeleton
import so3

PARENTS = skeleton.PARENTS


def animate_skeletons(motions, titles, path, fps=18):
    """motions: list of (trans (T,3), R (T,J,3,3)) tensors. Renders an animated skeleton row -> GIF."""
    pos = [skeleton.forward_kinematics(tr.cpu(), R.cpu()).numpy() for (tr, R) in motions]   # each (T,J,3)
    n, T = len(pos), pos[0].shape[0]
    allp = np.concatenate([p.reshape(-1, 3) for p in pos], 0)
    pad = 0.2
    lim = [(allp[:, d].min() - pad, allp[:, d].max() + pad) for d in range(3)]

    fig = plt.figure(figsize=(4.3 * n, 4.8))
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]

    def draw(fr):
        for ax, p, title in zip(axes, pos, titles):
            ax.cla()
            Pf = p[fr]
            for j, par in enumerate(PARENTS):
                if par >= 0:
                    ax.plot([Pf[par, 0], Pf[j, 0]], [Pf[par, 2], Pf[j, 2]], [Pf[par, 1], Pf[j, 1]],
                            color="#2e86de", lw=2.5)
            ax.scatter(Pf[:, 0], Pf[:, 2], Pf[:, 1], s=12, color="#c0392b")
            ax.set_xlim(lim[0]); ax.set_ylim(lim[2]); ax.set_zlim(lim[1])
            ax.set_box_aspect((1, 1, 1.6))
            ax.set_title(f"{title}   frame {fr + 1}/{T}", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return []

    FuncAnimation(fig, draw, frames=T, interval=1000 // fps).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)


def smoothness(R_prior, R_gen, R_data, path, tag):
    """Per-joint frame-to-frame geodesic step (rad): generated should overlap data, both far from prior.

    Two panels (full range + zoom). Data is a filled histogram; generated is a dashed line drawn ON TOP
    so the overlap is visible (otherwise the perfectly-matching curves hide each other).
    """
    dp = so3.geodesic_dist(R_prior[:, :-1], R_prior[:, 1:]).flatten().cpu().numpy()
    dg = so3.geodesic_dist(R_gen[:, :-1], R_gen[:, 1:]).flatten().cpu().numpy()
    dd = so3.geodesic_dist(R_data[:, :-1], R_data[:, 1:]).flatten().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    views = [(0.0, 3.2, "full range — prior is the broad noisy bump"),
             (0.0, 0.5, "zoom — generated (green) sits on data (black)")]
    for ax, (xlo, xhi, ttl) in zip(axes, views):
        bins = np.linspace(xlo, xhi, 80)
        ax.hist(dp, bins=bins, histtype="step", density=True, color="#999999", lw=1.6,
                label=f"prior  (mean {dp.mean():.3f})")
        ax.hist(dd, bins=bins, histtype="stepfilled", density=True, color="#222222", alpha=0.30,
                label=f"data  (mean {dd.mean():.3f})")
        ax.hist(dg, bins=bins, histtype="step", density=True, color="#27ae60", lw=2.2, linestyle="--",
                label=f"generated  (mean {dg.mean():.3f})")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("per-joint frame-to-frame geodesic step (rad)  — lower = smoother")
        ax.set_title(ttl, fontsize=11)
    axes[0].set_ylabel("density"); axes[1].legend(fontsize=9)
    fig.suptitle(f"[{tag}] motion smoothness — generated overlaps data, both far from the noisy prior", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print("wrote", path)
