"""
HumanML3D 263-dim features -> our representation (root translation R^3 + per-joint SO(3)^22).

263 layout (T2M / HumanML3D, 22 joints):
  [0]        root_rot_velocity (about the vertical/Y axis, per frame)
  [1:3]      root_linear_velocity (x, z) in the body-local frame
  [3]        root height (y)
  [4:67]     ric_data   — joint positions 1..21 relative to root (21*3 = 63)
  [67:193]   rot_data   — 6D rotations of joints 1..21 (21*6 = 126)
  [193:259]  local velocity (22*3 = 66)
  [259:263]  foot contacts (4)

Our model needs: root global rotation (Y-facing, from integrating rot_velocity), the 21 joint LOCAL
rotations (from rot_data 6D — same Gram-Schmidt convention as so3.sixd_to_matrix), and root translation
(from integrating the linear velocity rotated to world + the height channel).

The 6D->matrix convention matches HumanML3D's cont6d_to_matrix (verified: both are Gram-Schmidt columns).
"""
import os
import numpy as np
import torch

import so3

HML = os.environ.get("HML_DIR", "data/HumanML3D")
NUM_JOINTS = 22
_Y = torch.tensor([0.0, 1.0, 0.0])


def feat_to_rep(data):
    """263 feature (T,263) -> (trans (T,3), R (T,22,3,3))."""
    T = data.shape[0]
    # root facing angle = cumulative rotation velocity (shifted, like HumanML3D recover_root_rot_pos)
    ang = torch.zeros(T)
    ang[1:] = torch.cumsum(data[:-1, 0], dim=0)
    # HumanML3D's root quaternion is [cos(ang),0,sin(ang),0] = rotation about Y by 2*ang (not ang).
    R_root = so3.exp((2.0 * ang)[:, None] * _Y)                # (T,3,3) rotation about Y

    # root translation: local xz velocity -> world (rotate by R_root^{-1}, per HumanML3D), integrate
    vel = torch.zeros(T, 3)
    vel[1:, 0] = data[:-1, 1]
    vel[1:, 2] = data[:-1, 2]
    world_vel = torch.einsum("tij,tj->ti", R_root.transpose(-1, -2), vel)
    trans = torch.cumsum(world_vel, dim=0)
    trans[:, 1] = data[:, 3]

    R_joints = so3.sixd_to_matrix(data[:, 67:193].reshape(T, 21, 6))   # (T,21,3,3)
    R = torch.cat([R_root[:, None], R_joints], dim=1)                  # (T,22,3,3)
    return trans, R


def split_ids(split):
    return [l.strip() for l in open(os.path.join(HML, f"{split}.txt")) if l.strip()]


def build_windows(split="train", T=64, stride_frac=0.5, max_clips=20000, max_files=None):
    ids = split_ids(split)
    if max_files:
        ids = ids[:max_files]
    step = max(1, int(T * stride_frac))
    Ts, Rs = [], []
    for mid in ids:
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        if not os.path.exists(f):
            continue
        try:
            data = torch.from_numpy(np.load(f)).float()
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] != 263 or data.shape[0] < T:
            continue
        trans, R = feat_to_rep(data)
        for s in range(0, data.shape[0] - T + 1, step):
            Ts.append(trans[s:s + T] - trans[s:s + 1])     # translation relative to window start
            Rs.append(R[s:s + T])
            if len(Rs) >= max_clips:
                return torch.stack(Ts), torch.stack(Rs)
    if not Rs:
        raise RuntimeError("no HumanML3D windows built")
    return torch.stack(Ts), torch.stack(Rs)


def hml_fk(trans, R, off):
    """HumanML3D FK convention: joint j is placed by its OWN global rotation gr[j]=gr[parent]@R[j].
    trans (...,3), R (...,22,3,3), off (22,3) -> positions (...,22,3)."""
    import skeleton
    gr = [None] * NUM_JOINTS
    gp = [None] * NUM_JOINTS
    for j, p in enumerate(skeleton.PARENTS):
        if p < 0:
            gr[j] = R[..., j, :, :]
            gp[j] = trans
        else:
            gr[j] = gr[p] @ R[..., j, :, :]
            gp[j] = gp[p] + (gr[j] @ off[j].unsqueeze(-1)).squeeze(-1)
    return torch.stack(gp, dim=-2)


# HumanML3D's exact rest bone directions (paramUtil.t2m_raw_offsets); scaled by bone lengths.
T2M_RAW_OFFSETS = torch.tensor(
    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
     [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
     [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]], dtype=torch.float32)


def hml_offsets(nj):
    """Fixed HumanML3D rest offsets = t2m_raw_offset direction * bone length. All retargeted motions
    share the same bone lengths, so any clip's new_joints gives them."""
    import skeleton
    off = T2M_RAW_OFFSETS.clone()
    for j, p in enumerate(skeleton.PARENTS):
        if p >= 0:
            off[j] = T2M_RAW_OFFSETS[j] * (nj[0, j] - nj[0, p]).norm()
    return off


def _validate():
    os.makedirs("report", exist_ok=True)
    for mid in split_ids("train")[:3]:
        data = torch.from_numpy(np.load(os.path.join(HML, "new_joint_vecs", mid + ".npy"))).float()
        nj = torch.from_numpy(np.load(os.path.join(HML, "new_joints", mid + ".npy"))).float()
        trans, R = feat_to_rep(data)
        root_err = (trans - nj[:, 0]).abs().max().item()
        off = hml_offsets(nj)
        fk_err = (hml_fk(nj[:, 0], R, off) - nj).abs().max().item()
        print(f"{mid}: T={data.shape[0]:3d}  root_err={root_err:.4f}  FK-vs-ric_err={fk_err:.4f} (redundancy, ~0.1-0.2 ok)")

    mid = split_ids("train")[0]
    data = torch.from_numpy(np.load(os.path.join(HML, "new_joint_vecs", mid + ".npy"))).float()
    nj = torch.from_numpy(np.load(os.path.join(HML, "new_joints", mid + ".npy"))).float()
    trans, R = feat_to_rep(data)
    pos = hml_fk(trans, R, hml_offsets(nj)).numpy()
    _render_positions(pos, "report/hml_recon.gif")
    print("wrote report/hml_recon.gif")


def _render_positions(pos, path, fps=18):
    """pos (T,22,3) -> animated skeleton GIF (positions already FK'd)."""
    import numpy as _np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import skeleton
    T = pos.shape[0]
    allp = pos.reshape(-1, 3)
    pad = 0.2
    lim = [(allp[:, d].min() - pad, allp[:, d].max() + pad) for d in range(3)]
    fig = plt.figure(figsize=(4.5, 5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(fr):
        ax.cla()
        P = pos[fr]
        for j, par in enumerate(skeleton.PARENTS):
            if par >= 0:
                ax.plot([P[par, 0], P[j, 0]], [P[par, 2], P[j, 2]], [P[par, 1], P[j, 1]], color="#2e6fdb", lw=2.4)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[2]); ax.set_zlim(*lim[1])
        ax.set_box_aspect((1, 1, 1.6)); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"HumanML3D reconstruction  frame {fr + 1}/{T}", fontsize=10)
        return []

    FuncAnimation(fig, draw, frames=T, interval=1000 // fps).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    _validate()
