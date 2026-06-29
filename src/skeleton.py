"""
SMPL 22-joint skeleton: kinematic tree, approximate rest offsets, and forward kinematics (FK).

Used to turn (root translation, per-joint local rotations) into 3D joint positions for rendering
and for any position-space metrics. Rest offsets are an approximate humanoid T-pose (not the exact
SMPL betas=0 skeleton) — good enough for Phase A's synthetic motions and visualization.
"""
import torch

# SMPL kinematic tree, first 22 joints (hands dropped) — parent index per joint.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
JOINT_NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
               "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head",
               "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]
NUM_JOINTS = len(PARENTS)

# Approximate T-pose joint positions (meters). x = left, y = up, z = forward.
REST_POS = torch.tensor([
    [0.00, 0.00, 0.00],   # 0  pelvis
    [0.08, -0.06, 0.00],  # 1  l_hip
    [-0.08, -0.06, 0.00], # 2  r_hip
    [0.00, 0.12, 0.00],   # 3  spine1
    [0.09, -0.45, 0.00],  # 4  l_knee
    [-0.09, -0.45, 0.00], # 5  r_knee
    [0.00, 0.26, 0.00],   # 6  spine2
    [0.09, -0.85, 0.00],  # 7  l_ankle
    [-0.09, -0.85, 0.00], # 8  r_ankle
    [0.00, 0.38, 0.00],   # 9  spine3
    [0.10, -0.90, 0.12],  # 10 l_foot
    [-0.10, -0.90, 0.12], # 11 r_foot
    [0.00, 0.55, 0.00],   # 12 neck
    [0.07, 0.45, 0.00],   # 13 l_collar
    [-0.07, 0.45, 0.00],  # 14 r_collar
    [0.00, 0.66, 0.00],   # 15 head
    [0.17, 0.46, 0.00],   # 16 l_shoulder
    [-0.17, 0.46, 0.00],  # 17 r_shoulder
    [0.42, 0.46, 0.00],   # 18 l_elbow
    [-0.42, 0.46, 0.00],  # 19 r_elbow
    [0.66, 0.46, 0.00],   # 20 l_wrist
    [-0.66, 0.46, 0.00],  # 21 r_wrist
])


def rest_offsets():
    """Offset of each joint from its parent in the rest pose, (J,3)."""
    off = REST_POS.clone()
    for j, p in enumerate(PARENTS):
        off[j] = REST_POS[j] - REST_POS[p] if p >= 0 else torch.zeros(3)
    return off


def forward_kinematics(root_trans, R_local, offsets=None):
    """root_trans (...,3), R_local (...,J,3,3) -> global joint positions (...,J,3)."""
    if offsets is None:
        offsets = rest_offsets().to(R_local.device, R_local.dtype)
    gr, gp = [None] * NUM_JOINTS, [None] * NUM_JOINTS
    for j, p in enumerate(PARENTS):
        off = offsets[j]
        if p < 0:
            gr[j] = R_local[..., j, :, :]
            gp[j] = root_trans + off
        else:
            gr[j] = gr[p] @ R_local[..., j, :, :]
            gp[j] = gp[p] + (gr[p] @ off.unsqueeze(-1)).squeeze(-1)
    return torch.stack(gp, dim=-2)
