"""
Motion metrics for Phase A: a FID-style Frechet distance on pose/velocity features + jitter + a table.

We have no learned motion encoder for the synthetic data, so we use simple, interpretable features —
FK joint positions relative to the root, and their frame-to-frame velocities — and compute the
Frechet distance between the GENERATED set and the DATA set (mean+cov, the FID formula). Lower = closer
to the data distribution; achievable target ~0 (matching data). The prior gives the chance floor.
"""
import numpy as np
import torch
from scipy import linalg

import skeleton
import so3


def _frechet(feat_a, feat_b, eps=1e-6):
    """Fréchet (FID) distance between two feature sets, each (N, D), via the standard pytorch-fid
    formula: ||mu_a-mu_b||^2 + tr(Ca + Cb - 2 sqrt(Ca Cb)), with a proper matrix square root
    (scipy.linalg.sqrtm) and a diagonal-jitter fallback if the product is singular."""
    a = feat_a.detach().cpu().numpy().astype(np.float64)
    b = feat_b.detach().cpu().numpy().astype(np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean = linalg.sqrtm(ca @ cb)
    if isinstance(covmean, tuple):                           # older scipy returns (sqrt, errest)
        covmean = covmean[0]
    if not np.isfinite(covmean).all():                       # singular product -> jitter the diagonal
        off = np.eye(ca.shape[0]) * eps
        covmean = linalg.sqrtm((ca + off) @ (cb + off))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(ca) + np.trace(cb) - 2.0 * np.trace(covmean))


def _pose_features(trans, R):
    """Per-frame joint positions relative to the pelvis (pose, translation-invariant). (B*T, J*3)."""
    pos = skeleton.forward_kinematics(trans, R)
    pos = pos - pos[:, :, :1, :]
    B, T, J, _ = pos.shape
    return pos.reshape(B * T, J * 3)


def _vel_features(trans, R):
    """Per-frame velocity of joint positions (dynamics). (B*(T-1), J*3)."""
    pos = skeleton.forward_kinematics(trans, R)
    vel = pos[:, 1:] - pos[:, :-1]
    B, T1, J, _ = vel.shape
    return vel.reshape(B * T1, J * 3)


def rot_jitter(R):
    return so3.geodesic_dist(R[:, :-1], R[:, 1:]).mean().item()


def trans_jitter(trans):
    return (trans[:, 1:] - trans[:, :-1]).norm(dim=-1).mean().item()


def compute(gen, prior, data):
    """gen/prior/data each = (trans, R). Returns a metrics dict."""
    (tg, Rg), (tp, Rp), (td, Rd) = gen, prior, data
    return {
        "pose_fid_gen": _frechet(_pose_features(tg, Rg), _pose_features(td, Rd)),
        "pose_fid_prior": _frechet(_pose_features(tp, Rp), _pose_features(td, Rd)),
        "vel_fid_gen": _frechet(_vel_features(tg, Rg), _vel_features(td, Rd)),
        "vel_fid_prior": _frechet(_vel_features(tp, Rp), _vel_features(td, Rd)),
        "rot_jitter_gen": rot_jitter(Rg), "rot_jitter_data": rot_jitter(Rd), "rot_jitter_prior": rot_jitter(Rp),
        "trans_jitter_gen": trans_jitter(tg), "trans_jitter_data": trans_jitter(td), "trans_jitter_prior": trans_jitter(tp),
        "on_manifold_err": (Rg.transpose(-1, -2) @ Rg - torch.eye(3, device=Rg.device)).abs().max().item(),
    }


def table_md(m, method):
    rows = [
        ("pose-FID ↓",       m["pose_fid_prior"],    m["pose_fid_gen"],    0.0),
        ("velocity-FID ↓",   m["vel_fid_prior"],     m["vel_fid_gen"],     0.0),
        ("rot jitter (rad)",      m["rot_jitter_prior"],  m["rot_jitter_gen"],  m["rot_jitter_data"]),
        ("trans jitter (m)",      m["trans_jitter_prior"], m["trans_jitter_gen"], m["trans_jitter_data"]),
    ]
    lines = [f"### Motion metrics — {method}", "",
             "| metric | prior (chance) | **generated (ours)** | data (target) |",
             "|---|---|---|---|"]
    for name, p, g, d in rows:
        lines.append(f"| {name} | {p:.4f} | **{g:.4f}** | {d:.4f} |")
    lines.append(f"| on-manifold err | — | {m['on_manifold_err']:.1e} | 0 |")
    lines += [
        "",
        "*pose-/velocity-FID = Fréchet distance between generated and data features (FK joint "
        "positions / velocities). Goal: drive generated→data (→0); prior is the chance floor.*",
        "",
        "**Real text-to-motion SOTA reference (HumanML3D — the Phase B/C target):** RMG FID **0.043**, "
        "MoMask 0.045, MotionGPT 0.232, MLD 0.473. (Different scale from our synthetic FID above; listed as "
        "the goalpost for when we move to real data.)",
    ]
    return "\n".join(lines)
