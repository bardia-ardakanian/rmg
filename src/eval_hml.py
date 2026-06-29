"""
Held-out HumanML3D evaluation of a VFM model, on the CORRECT HumanML3D FK geometry.

Two fixes over the generic eval_real.py:
  1. Features use HumanML3D's EXACT forward_kinematics_cont6d (own-rotation convention, ref skel 000021),
     not our approximate SMPL FK — so the Fréchet lives in the data's true joint space.
  2. The real reference is the held-out TEST split (build_windows split='test'), not the training cache.

This is our INTERNAL geometric metric (interpretable, fast). The canonical Guo et al. T2M benchmark
(FID / R-precision / MM-Dist / Diversity on the 263 features) is computed separately in eval_official.py.

    python eval_hml.py --ckpt runs/hml_vfm/model.pth --n 128 --n_real 512 --out report
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.environ.get("HML3D_REPO", os.path.expanduser("~/hml3d_repo")))
import numpy as np
np.float = float  # numpy-2 compat for HumanML3D code
import torch

import paramUtil
from common.skeleton import Skeleton

import hml_data
import metrics

HML = os.environ.get("HML_DIR", "$HML_DIR")

_skel = Skeleton(torch.tensor(paramUtil.t2m_raw_offsets).float(), paramUtil.t2m_kinematic_chain, "cpu")
_skel.set_offset(_skel.get_offsets_joints(torch.from_numpy(np.load(f"{HML}/new_joints/000021.npy")).float()[0]))


def positions(trans, R):
    """(T,3),(T,22,3,3) -> (T,22,3) world joint positions via HumanML3D's exact FK."""
    cont6d = torch.cat([R[..., 0], R[..., 1]], dim=-1)
    return _skel.forward_kinematics_cont6d(cont6d, trans).detach()


def _all_pos(trans, R):
    return torch.stack([positions(trans[i], R[i]) for i in range(trans.shape[0])])  # (N,T,22,3)


def pose_feats(P):
    P = (P - P[:, :, :1])[:, :, 1:]                  # relative to pelvis, drop the (now-zero) pelvis column
    N, T, J, _ = P.shape
    return P.reshape(N * T, J * 3)


def vel_feats(P):
    V = P[:, 1:] - P[:, :-1]
    N, T1, J, _ = V.shape
    return V.reshape(N * T1, J * 3)
