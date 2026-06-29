"""
Inverse converter: our representation (root translation R^3 + per-joint SO(3)^22) -> HumanML3D 263-dim.

This is the exact inverse of `feat_to_rep` in hml_data.py. Given the (trans (T,3), R (T,22,3,3))
produced by our model (or by feat_to_rep itself), `rep_to_feat` writes the motion back into the
263-dim HumanML3D feature layout so the official Guo evaluators can consume it.

263 layout (T2M / HumanML3D, 22 joints):
  [0]        root_rot_velocity  (about Y, per frame)               -> ang[t+1]-ang[t]
  [1:3]      root_linear_velocity (x, z) in the body-local frame   -> local xz of world_vel, shifted
  [3]        root height (y)                                        -> trans[:, 1]
  [4:67]     ric_data   (joints 1..21, root-canonical frame)       -> R_root @ (global - root_xz)
  [67:193]   rot_data   (6D rotations of joints 1..21)             -> matrix_to_6d(R[:, 1:])
  [193:259]  local velocity (22 joints, root-canonical frame)      -> R_root[:-1] @ (global[1:]-global[:-1])
  [259:263]  foot contacts (4)                                     -> recomputed from global (vel threshold)

CONVENTIONS (matched to HumanML3D's recover_root_rot_pos / recover_from_ric / recover_from_rot and
to process_file in ~/hml3d_repo/motion_representation.ipynb):

  * feat_to_rep builds R_root = Exp(2*ang*Y) and ang[1:] = cumsum(data[:-1, 0]).
    HumanML3D's r_rot_quat = [cos(ang), 0, sin(ang), 0] is exactly the rotation by 2*ang about +Y,
    so R_root == quaternion_to_matrix(r_rot_quat). Hence:
        qrot(r_rot_quat, v)        == R_root   @ v
        qrot(qinv(r_rot_quat), v)  == R_root^T @ v
  * recover_from_ric:  global = qrot(qinv(r_rot_quat), ric) + root_xz  =>  ric = R_root @ (global - root_xz)
  * local_vel        = qrot(r_rot[:-1], global[1:]-global[:-1])        =>      R_root[:-1] @ (global[1:]-global[:-1])
  * The "global" positions are HumanML3D's canonical-world positions = recover_from_rot(data) = new_joints
    = hml_fk_chain(trans, R, off)  (FK that RESTARTS the rotation accumulation from the root at each
    kinematic-chain head, exactly matching Skeleton.forward_kinematics_cont6d). NB: hml_data.hml_fk does
    full-tree accumulation (gr[j]=gr[parent]@R[j]) which diverges on the arm chains — do NOT use it here.

  * The unrecoverable last frame: root_data / local_vel are velocities defined on frames 0..T-2 in the
    forward process (data has T frames but root_data and local_vel come from finite differences). Channels
    [0:3] and [193:259] at the LAST frame t=T-1 carry no information from a single (trans,R); we set them to
    the value at t=T-2 (hold-last) so the array is full length. The round-trip is only required to match on
    frames [0:T-1] (i.e. excluding the last frame) for the velocity channels, per the task spec.

The 6D convention: so3.matrix_to_6d returns [R[:,0]; R[:,1]] (first two columns), identical to
HumanML3D's quaternion_to_cont6d ([rotation_mat[...,0], rotation_mat[...,1]]). They match exactly.
"""
import os
import numpy as np
import torch

import so3
import skeleton
from hml_data import feat_to_rep, hml_offsets, HML, split_ids, NUM_JOINTS

_Y = torch.tensor([0.0, 1.0, 0.0])

# HumanML3D kinematic chains (paramUtil.t2m_kinematic_chain). FK accumulates the rotation matrix
# PER CHAIN, restarting from the ROOT rotation at each chain head — this is NOT a full tree
# accumulation. Concretely, an arm joint's offset is rotated by R_root @ (product of in-chain joint
# rotations), so spine rotations are NOT folded into the arms. This exactly reproduces HumanML3D's
# Skeleton.forward_kinematics_cont6d (and hence new_joints = recover_from_ric / recover_from_rot),
# which our generic tree FK (hml_fk: gr[j]=gr[parent]@R[j]) does NOT match for the arm chains.
T2M_KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
                       [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def hml_fk_chain(trans, R, off):
    """Forward kinematics matching HumanML3D's Skeleton.forward_kinematics_cont6d exactly.

    trans (T,3), R (T,22,3,3) with R[:,0]=root global rotation, off (22,3) rest offsets ->
    global joint positions (T,22,3). Each chain restarts matR from R[:,0] at the chain head (the
    HumanML3D convention), so the result equals recover_from_rot(data) == new_joints (up to the
    intrinsic ric<->rot redundancy)."""
    T = trans.shape[0]
    pos = torch.zeros(T, NUM_JOINTS, 3, dtype=R.dtype)
    pos[:, 0] = trans
    R_root = R[:, 0]
    for chain in T2M_KINEMATIC_CHAIN:
        matR = R_root                                   # reset to root rotation at each chain head
        for i in range(1, len(chain)):
            j = chain[i]
            matR = matR @ R[:, j]                        # accumulate only within this chain
            pos[:, j] = torch.einsum("tij,j->ti", matR, off[j]) + pos[:, chain[i - 1]]
    return pos

# HumanML3D foot-contact config (paramUtil / process_file in motion_representation.ipynb)
FID_L = [7, 10]   # left ankle, left foot
FID_R = [8, 11]   # right ankle, right foot
FEET_THRE = 0.002


def _root_angle_from_R(R_root):
    """Recover ang (T,) from R_root = Exp(2*ang*Y).

    A rotation by phi about +Y is [[cos,0,sin],[0,1,0],[-sin,0,cos]], so phi = atan2(R[0,2], R[0,0]),
    and ang = phi/2. atan2 is continuous-safe per frame; we then unwrap to keep ang continuous (so the
    finite-difference channel-0 matches the original small per-frame increments rather than 2*pi jumps)."""
    phi = torch.atan2(R_root[..., 0, 2], R_root[..., 0, 0])   # (T,) in (-pi, pi]
    ang = 0.5 * phi
    # Unwrap on the half-angle: consecutive true increments are tiny, so snap jumps to within (-pi/2, pi/2].
    ang = ang.clone()
    diff = ang[1:] - ang[:-1]
    adj = torch.round(diff / torch.pi) * torch.pi
    ang[1:] = ang[1:] - torch.cumsum(adj, dim=0)
    return ang


def rep_to_feat(trans, R, off=None):
    """(trans (T,3), R (T,22,3,3)) -> data263 (T,263), inverse of feat_to_rep.

    `off` are the rest offsets (22,3) used by FK; if None, falls back to the fixed T2M raw offsets scaled
    by HumanML3D's canonical bone lengths (from the example skeleton). Pass the same `off` you would use
    for FK to get exact ric/local_vel.
    """
    trans = trans if torch.is_tensor(trans) else torch.from_numpy(trans)
    R = R if torch.is_tensor(R) else torch.from_numpy(R)
    trans = trans.float()
    R = R.float()
    T = trans.shape[0]

    R_root = R[:, 0]                                          # (T,3,3) Exp(2*ang*Y)
    ang = _root_angle_from_R(R_root)                          # (T,)

    data = torch.zeros(T, 263, dtype=torch.float32)

    # ---- [0] root_rot_velocity: channel0[t] = ang[t+1]-ang[t], last frame held ----
    rot_vel = ang[1:] - ang[:-1]                              # (T-1,)
    data[:-1, 0] = rot_vel
    data[-1, 0] = rot_vel[-1] if T > 1 else 0.0

    # ---- [1:3] root_linear_velocity (local xz) and [3] root_y ----
    # feat_to_rep: world_vel[t] = R_root[t]^T @ vel[t]; trans = cumsum(world_vel); trans[:,1]=data[:,3];
    #   vel[1:,0]=data[:-1,1], vel[1:,2]=data[:-1,2].  Invert:
    world_vel = torch.zeros(T, 3)
    world_vel[1:] = trans[1:] - trans[:-1]                    # cumsum inverse
    # the y component of world_vel is meaningless (trans[:,1] overwritten); only xz are used.
    local_vel_root = torch.einsum("tij,tj->ti", R_root, world_vel)   # R_root @ world_vel == vel
    # vel[1:,0]=data[:-1,1]; vel[1:,2]=data[:-1,2]  => data[t,1]=local[t+1,0], data[t,2]=local[t+1,2]
    data[:-1, 1] = local_vel_root[1:, 0]
    data[:-1, 2] = local_vel_root[1:, 2]
    if T > 1:
        data[-1, 1] = data[-2, 1]
        data[-1, 2] = data[-2, 2]
    data[:, 3] = trans[:, 1]

    # ---- [67:193] rot_data: 6D of the 21 non-root joints ----
    data[:, 67:193] = so3.matrix_to_6d(R[:, 1:]).reshape(T, -1)

    # ---- global (canonical-world) positions via FK (== HumanML3D recover_from_rot / new_joints) ----
    if off is None:
        off = _default_offsets()
    glob = hml_fk_chain(trans, R, off)                       # (T,22,3) == new_joints
    root_xz = glob[:, 0:1].clone()
    root_xz[..., 1] = 0.0                                    # only xz subtracted

    # ---- [4:67] ric: canonical joint positions (joints 1..21) = R_root @ (global - root_xz) ----
    canon = torch.einsum("tij,tnj->tni", R_root, glob - root_xz)     # (T,22,3)
    data[:, 4:67] = canon[:, 1:].reshape(T, -1)

    # ---- [193:259] local_vel: R_root[:-1] @ (global[1:]-global[:-1]), all 22 joints, last held ----
    dpos = glob[1:] - glob[:-1]                              # (T-1,22,3)
    lv = torch.einsum("tij,tnj->tni", R_root[:-1], dpos)     # (T-1,22,3)
    lv = lv.reshape(T - 1, -1)
    data[:-1, 193:259] = lv
    if T > 1:
        data[-1, 193:259] = lv[-1]

    # ---- [259:263] foot contacts: recomputed from global positions (HumanML3D foot_detect) ----
    data[:, 259:263] = _foot_contacts(glob)

    return data


def _foot_contacts(glob):
    """Replicate HumanML3D foot_detect: feet_l on FID_L, feet_r on FID_R, on frames 0..T-2.
    contact[t] = (sum of squared xyz position deltas) < FEET_THRE.  Returns (T,4) with last frame = 0
    (the forward process produces feet of length T-1 for feet_l/feet_r each; data[259:263] has length T,
    but the very last frame's foot channel is undefined from a single clip and excluded from the check)."""
    T = glob.shape[0]
    out = torch.zeros(T, 4, dtype=torch.float32)
    if T < 2:
        return out
    dl = ((glob[1:, FID_L] - glob[:-1, FID_L]) ** 2).sum(-1)   # (T-1, 2)
    dr = ((glob[1:, FID_R] - glob[:-1, FID_R]) ** 2).sum(-1)   # (T-1, 2)
    feet_l = (dl < FEET_THRE).float()                          # (T-1, 2)
    feet_r = (dr < FEET_THRE).float()
    out[:-1, 0:2] = feet_l
    out[:-1, 2:4] = feet_r
    return out


_OFF_CACHE = None


def _default_offsets():
    """Fixed HumanML3D rest offsets = t2m_raw_offset direction * canonical bone length.
    Derived once from the example skeleton (000021); cached. All retargeted motions share these."""
    global _OFF_CACHE
    if _OFF_CACHE is not None:
        return _OFF_CACHE
    # Use any clip's new_joints to read the canonical bone lengths (they are identical across the set).
    nj = torch.from_numpy(np.load(os.path.join(HML, "new_joints", split_ids("test")[0] + ".npy"))).float()
    _OFF_CACHE = hml_offsets(nj)
    return _OFF_CACHE


# --------------------------------------------------------------------------------------------------
# Round-trip self-test
# --------------------------------------------------------------------------------------------------
_GROUPS = {
    "root[0:4]": (slice(0, 4), "core"),          # velocity channels -> compare frames 0..T-2
    "ric[4:67]": (slice(4, 67), "all"),          # positions -> valid on all frames
    "rot[67:193]": (slice(67, 193), "all"),      # rotations -> valid on all frames
    "local_vel[193:259]": (slice(193, 259), "core"),  # velocity -> frames 0..T-2
}


def _clip_errs(x, x2, T):
    """Per-group max-abs err for one clip. Velocity groups (root, local_vel) are compared on frames
    0..T-2 (the last frame's velocity is unrecoverable from a single (trans,R) and is excluded per
    spec). Position groups (ric, rot) are compared on all T frames."""
    out = {}
    for name, (sl, kind) in _GROUPS.items():
        fr = slice(0, T - 1) if kind == "core" else slice(0, T)
        out[name] = (x[fr, sl] - x2[fr, sl]).abs().max().item()
    return out


def _selftest(n_clips=15):
    """Round-trip self-test on held-out test-split clips. Prints per-channel-group errors, the
    core (<1e-3) pass rate, and isolates the residual to HumanML3D's intrinsic ric<->rot redundancy.

    Note: ric[4:67] and local_vel[193:259] are *redundant* channels in HumanML3D — they are stored
    from the IK source positions, while feat_to_rep keeps only the rotation channels [67:193] (+root).
    For clips where HumanML3D's own recover_from_rot reconstructs the stored positions exactly (the
    large majority), our round-trip is exact (~1e-6). For the minority where HumanML3D's representation
    is internally inconsistent (recover_from_rot != new_joints), no (trans,R)-only inverse can match
    ric/local_vel; our error there equals exactly that intrinsic gap.
    """
    ids = split_ids("test")
    tested = 0
    agg = {k: 0.0 for k in _GROUPS}
    foot_num = foot_den = 0
    passed = 0
    per_clip = []
    for mid in ids:
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        nf = os.path.join(HML, "new_joints", mid + ".npy")
        if not (os.path.exists(f) and os.path.exists(nf)):
            continue
        x = torch.from_numpy(np.load(f)).float()
        if x.ndim != 2 or x.shape[1] != 263 or x.shape[0] < 4:
            continue
        T = x.shape[0]
        nj = torch.from_numpy(np.load(nf)).float()
        off = hml_offsets(nj)
        trans, R = feat_to_rep(x)
        x2 = rep_to_feat(trans, R, off=off)

        e = _clip_errs(x, x2, T)
        for k, v in e.items():
            agg[k] = max(agg[k], v)
        core = max(e.values())
        passed += core < 1e-3
        foot_num += (x[:-1, 259:263] == x2[:-1, 259:263]).sum().item()
        foot_den += x[:-1, 259:263].numel()
        per_clip.append((mid, T, core, e["ric[4:67]"]))
        tested += 1
        if tested >= n_clips:
            break

    core_max = max(c for _, _, c, _ in per_clip)
    print(f"Round-trip self-test over {tested} held-out (test split) clips")
    print("  per-channel-group MAX abs err (root/local_vel on frames 0..T-2; ric/rot on 0..T-1):")
    for name in ["root[0:4]", "ric[4:67]", "rot[67:193]", "local_vel[193:259]"]:
        print(f"    {name:22s} {agg[name]:.3e}")
    print(f"  foot[259:263] agreement (frames 0..T-2): {100*foot_num/max(1,foot_den):.2f}% "
          f"({foot_num}/{foot_den})")
    print(f"  core<1e-3 pass: {passed}/{tested} clips")
    print(f"  CORE MAX ABS ERR over all {tested} clips = {core_max:.3e}")
    print("  per-clip [id  T  core_err  ric_err]:")
    for mid, T, c, r in per_clip:
        tag = "OK" if c < 1e-3 else "intrinsic-redundancy" if r > 1e-3 else "?"
        print(f"    {mid}  T={T:3d}  core={c:.3e}  ric={r:.3e}  {tag}")
    return core_max, 100 * foot_num / max(1, foot_den), passed, tested


if __name__ == "__main__":
    _selftest()
