"""Generate motions from the RMG model for a few prompts and export the GENERATED JOINT POSITIONS
(HumanML3D FK, 22 joints) for SMPL-X fitting -> mesh rendering."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # eval_hml.positions (HumanML3D FK)
import numpy as np
import torch

import s3
import qwen_text
from eval_hml import positions
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow

PROMPTS = [
    "a person walks forward",
    "a person sits down",
    "a person is jumping",
    "a person waves with the right hand",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rmg_base/model.pth")
    ap.add_argument("--L", type=int, default=120)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--weights", choices=["ema", "raw"], default="raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report/mesh_joints.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    c = torch.load(a.ckpt, map_location=dev)
    model = RMGTransformer(**c["config"]).to(dev)
    key = "state_dict" if a.weights == "raw" else "ema_state_dict"
    model.load_state_dict(c.get(key, c["state_dict"])); model.eval()
    flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))

    torch.manual_seed(a.seed)
    cond = qwen_text.encode(PROMPTS, device=dev).to(dev)
    tr, q = flow.sample(model, len(PROMPTS), a.L, text=cond, guidance=a.guidance, n_steps=a.steps, device=dev)
    tr, q = tr.cpu(), q.cpu()
    R = s3.quat_to_matrix(q)                                          # (P,L,22,3,3)
    joints = torch.stack([positions(tr[p], R[p]) for p in range(len(PROMPTS))])   # (P,L,22,3) HumanML3D FK
    np.savez(a.out, prompts=np.array(PROMPTS), joints=joints.numpy(), step=c.get("step", -1))
    print("wrote", a.out, "| joints", tuple(joints.shape), "| step", c.get("step", "?"))


if __name__ == "__main__":
    main()
