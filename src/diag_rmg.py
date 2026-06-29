"""Diagnose the flat-FID/chance-R-precision bug: is the text signal distinct, and does the model use it?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import s3
import qwen_text
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow

dev = "cuda" if torch.cuda.is_available() else "cpu"
caps = ["a person walks forward", "a person sits down", "a person runs quickly",
        "a person waves their right hand", "a person jumps high", "a person crawls on the floor"]

# 1) Qwen embedding distinctness
e = qwen_text.encode(caps, device=dev)                       # (6,1024) normalized
sim = (e @ e.T).numpy()
off = (sim.sum() - 6) / 30
print("=== Qwen distinctness ===")
print(np.round(sim, 2))
print(f"mean off-diagonal cosine sim: {off:.3f}  (want clearly < 0.9 so captions are distinguishable)")

# 2) Does the model actually respond to text? (same prior noise, different caption)
c = torch.load("runs/rmg_base/model_20k.pth", map_location=dev)
model = RMGTransformer(**c["config"]).to(dev)
model.load_state_dict(c.get("ema_state_dict", c["state_dict"])); model.eval()
flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))
L = 100
e = e.to(dev)


@torch.no_grad()
def gen(pt, pq, text, g):
    trans, quats = pt.clone(), pq.clone()
    h = 1 / 100
    for k in range(100):
        tv = torch.full((1,), k * h, device=dev)
        x = flow.pack(trans, quats)
        if text is not None and g != 1.0:
            pc = model(x, tv, text=text); pu = model(x, tv, text=None); pred = pu + g * (pc - pu)
        else:
            pred = model(x, tv, text=text)
        ptr, pqu = flow.unpack(pred)
        trans = trans + h * ptr; quats = s3.exp(quats, h * s3.proj(quats, pqu))
    return trans, quats


torch.manual_seed(0)
p0t, p0q = flow.prior(1, L, dev)
p1t, p1q = flow.prior(1, L, dev)
a_t, a_q = gen(p0t, p0q, e[0:1], 6.5)        # caption 0, noise 0
b_t, b_q = gen(p0t, p0q, e[1:2], 6.5)        # caption 1, noise 0  (only text differs)
c_t, c_q = gen(p1t, p1q, e[0:1], 6.5)        # caption 0, noise 1  (only noise differs)
print("\n=== model text-sensitivity (step 20k) ===")
print(f"diff(caption)  | same noise, cap0 vs cap1 : quat {(a_q-b_q).abs().mean():.4f}  trans {(a_t-b_t).abs().mean():.4f}")
print(f"diff(noise)    | same cap,  noise0 vs noise1: quat {(a_q-c_q).abs().mean():.4f}  trans {(a_t-c_t).abs().mean():.4f}")
print("-> if diff(caption) << diff(noise), the model is ignoring text.")

# 3) is the conditional velocity even different from unconditional at t=0?
x = flow.pack(p0t, p0q); t0 = torch.zeros(1, device=dev)
with torch.no_grad():
    vc = model(x, t0, text=e[0:1]); vu = model(x, t0, text=None)
print(f"\n||v_cond - v_uncond|| at t=0: {(vc-vu).abs().mean():.5f}   ||v_uncond||: {vu.abs().mean():.5f}")
