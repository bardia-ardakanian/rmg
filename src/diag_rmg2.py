"""Localize the generation bug: is the ODE transporting prior->data, or stuck near the prior?
Measures quaternion angle-from-identity (prior / generated / data) and FID (prior / generated) vs real."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.float = float
import torch

import s3
from rmg_to_263 import rmg_to_263
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow
from eval_official import OfficialEvaluator, prep_motion, collect_motion_embeddings, split_ids, HML, DIM_POSE
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = 256, 120
ev = OfficialEvaluator(dev)


def ang(q):
    return torch.arccos(q[..., 0].clamp(-1, 1)).mean().item()


def emb_of(trans, quats):
    reps = [rmg_to_263(trans[i], quats[i], off=None).numpy().astype(np.float32) for i in range(trans.shape[0])]
    m, l = zip(*[prep_motion(ev, r) for r in reps])
    return collect_motion_embeddings(ev, torch.from_numpy(np.stack(m)), torch.tensor(l), drop_last=False)


# real reference embeddings (full clips, cropped to mult of 4)
rm, rl = [], []
for mid in split_ids("test"):
    f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
    if not os.path.exists(f):
        continue
    x = np.load(f).astype(np.float32)
    if x.ndim != 2 or x.shape[1] != DIM_POSE or x.shape[0] < 64:
        continue
    T = (x.shape[0] // 4) * 4
    mp, ml = prep_motion(ev, x[:T]); rm.append(mp); rl.append(ml)
    if len(rm) >= N:
        break
real_emb = collect_motion_embeddings(ev, torch.from_numpy(np.stack(rm)), torch.tensor(rl), drop_last=False)
mur, cr = calculate_activation_statistics(real_emb)

c = torch.load("runs/rmg_base/model_20k.pth", map_location=dev)
flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))


def fid(emb):
    mu, cv = calculate_activation_statistics(emb)
    return float(calculate_frechet_distance(mur, cr, mu, cv))


# prior samples (no model)
torch.manual_seed(0)
pt, pq = flow.prior(N, L, dev)
prior_emb = emb_of(pt.cpu(), pq.cpu())
print(f"PRIOR   quat-angle={ang(pq):.3f}  FID={fid(prior_emb):.2f}")

# generated (ema and raw)
for tag, key in [("EMA", "ema_state_dict"), ("RAW", "state_dict")]:
    model = RMGTransformer(**c["config"]).to(dev)
    model.load_state_dict(c[key]); model.eval()
    torch.manual_seed(0)
    tr, q = flow.sample(model, N, L, guidance=1.0, n_steps=100, device=dev)   # unconditional, null text
    e = emb_of(tr.cpu(), q.cpu())
    np.random.seed(0)
    print(f"GEN[{tag}] quat-angle={ang(q):.3f}  FID={fid(e):.2f}  Div={calculate_diversity(e,200):.2f}")

# data reference angle
d = torch.load("cache_rmg_train.pt")
qa = []
for i in range(min(500, d['quats'].shape[0])):
    n = int(d['lengths'][i]); qa.append(d['quats'][i, :n])
print(f"DATA    quat-angle={ang(torch.cat(qa)):.3f}   real FID=0  (target)")
