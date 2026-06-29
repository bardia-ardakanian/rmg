"""
Convert RMG representation (trans R^3 + 22 S^3 quaternions) -> HumanML3D 263 for the official evaluator.
Thin wrapper: quaternions -> rotation matrices -> motion_real.rep_to_feat (verified SO(3)->263 converter).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import s3
from rep_to_feat import rep_to_feat


def rmg_to_263(trans, quats, off=None):
    """(T,3),(T,22,4) -> (T,263). off=None uses canonical bone lengths (for generated motion)."""
    R = s3.quat_to_matrix(quats)
    return rep_to_feat(trans, R, off=off)


if __name__ == "__main__":
    import statistics
    import hml_data
    from hml_data import feat_to_rep, hml_offsets, split_ids, HML
    errs = []
    for mid in split_ids("test"):
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        nf = os.path.join(HML, "new_joints", mid + ".npy")
        if not (os.path.exists(f) and os.path.exists(nf)):
            continue
        x = torch.from_numpy(np.load(f)).float()
        if x.ndim != 2 or x.shape[1] != 263 or x.shape[0] < 4:
            continue
        nj = torch.from_numpy(np.load(nf)).float()
        trans, R = feat_to_rep(x)
        q = s3.canonical(s3.matrix_to_quat(R))
        x2 = rmg_to_263(trans, q, off=hml_offsets(nj))
        T = x.shape[0]
        errs.append((x[:T - 1, :259] - x2[:T - 1, :259]).abs().max().item())
        if len(errs) >= 40:
            break
    print(f"rmg->263 round-trip core[0:259] err over {len(errs)} clips: "
          f"max={max(errs):.2e}  median={statistics.median(errs):.2e}")
