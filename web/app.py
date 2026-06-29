"""Web server for the RMG demo. POST a prompt, get back the generated joints (for the skeleton) and a
SMPL-X body mesh fit to those joints (for the body), and the browser renders both in 3D.

    python web/app.py            # serves on 0.0.0.0:8000  (tunnel: ssh -L 8000:localhost:8000 <host>)
"""
import base64
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "..", "src"), os.path.join(HERE, ".."),
          os.environ.get("MOTION_REAL", os.path.expanduser("~/rmg/motion_real"))]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

import numpy as np
import torch
import smplx
from flask import Flask, request, jsonify, send_from_directory

import s3
import qwen_text
from eval_hml import positions
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CKPT = os.environ.get("RMG_CKPT", os.path.join(HERE, "..", "runs", "rmg_base", "model.pth"))
SMPLX_PATH = os.environ.get("SMPLX_PATH", os.path.expanduser("~/smplx/models"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"loading {CKPT} on {DEV} ...")
c = torch.load(CKPT, map_location=DEV)
model = RMGTransformer(**c["config"]).to(DEV)
model.load_state_dict(c.get("ema_state_dict", c["state_dict"])); model.eval()
flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))
bm = smplx.create(SMPLX_PATH, model_type="smplx", gender="neutral", ext="npz",
                  use_pca=False, flat_hand_mean=True, batch_size=1).to(DEV)
FACES = bm.faces.astype(np.uint32)
_BODY_CACHE = {}


def body_model(L):
    """SMPL-X model sized for a batch of L frames (all default pose params then match the batch)."""
    if L not in _BODY_CACHE:
        _BODY_CACHE[L] = smplx.create(SMPLX_PATH, model_type="smplx", gender="neutral", ext="npz",
                                      use_pca=False, flat_hand_mean=True, batch_size=L).to(DEV)
    return _BODY_CACHE[L]


print("ready.")


def smooth(J, win):
    if not win or win <= 1:
        return J
    win = win + 1 if win % 2 == 0 else win          # kernel must be odd to keep the frame count
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = J.shape; jf = J.reshape(sh[0], -1)
    jp = np.pad(jf, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(jp[:, i], k, mode="valid") for i in range(jf.shape[1])], 1).reshape(sh)


def fit_smplx(target_np, iters=300):
    """Fit SMPL-X so its first 22 joints match the generated joints; return vertices (L,V,3)."""
    target = torch.tensor(target_np, dtype=torch.float32, device=DEV)
    L = target.shape[0]
    m = body_model(L)
    go = torch.zeros(L, 3, device=DEV, requires_grad=True)
    bp = torch.zeros(L, 63, device=DEV, requires_grad=True)
    transl = target[:, 0].clone().detach().requires_grad_(True)
    betas = torch.zeros(1, 10, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([go, bp, transl, betas], lr=0.05)
    for it in range(iters):
        if it == int(iters * 0.6):
            for g in opt.param_groups:
                g["lr"] = 0.01
        out = m(global_orient=go, body_pose=bp, transl=transl, betas=betas.expand(L, -1))
        data = ((out.joints[:, :22] - target) ** 2).sum(-1).mean()
        sm = ((bp[1:] - bp[:-1]) ** 2).mean() + 0.1 * ((transl[1:] - transl[:-1]) ** 2).mean()
        loss = data + 0.05 * sm + 5e-4 * (bp ** 2).mean() + 1e-3 * (betas ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = m(global_orient=go, body_pose=bp, transl=transl, betas=betas.expand(L, -1))
    return out.vertices.detach().cpu().numpy().astype(np.float32)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    d = request.get_json(force=True)
    prompt = (d.get("prompt") or "a person walks forward").strip()
    guidance = float(d.get("guidance", 6.5))
    L = max(40, min(196, int(d.get("length", 120))))
    win = int(d.get("smooth", 9))
    want_mesh = bool(d.get("mesh", True))
    torch.manual_seed(int(d.get("seed", 0)))

    with torch.no_grad():
        cond = qwen_text.encode([prompt], device=DEV).to(DEV)
        tr, q = flow.sample(model, 1, L, text=cond, guidance=guidance, n_steps=100, device=DEV)
        P = positions(tr[0].cpu(), s3.quat_to_matrix(q[0].cpu())).numpy()
    P = smooth(P, win).astype(np.float32)

    resp = {"parents": PARENTS, "fps": 20, "prompt": prompt}
    if want_mesh:
        verts = fit_smplx(P)                                   # (L,V,3), same frame as the joints
        floor = float(verts[:, :, 1].min())
        xz = P[0:1, 0:1, [0, 2]].copy()
        P = P.copy(); P[:, :, 1] -= floor; P[:, :, [0, 2]] -= xz
        verts[:, :, 1] -= floor; verts[:, :, [0, 2]] -= xz
        resp.update(verts=b64(verts), faces=b64(FACES), V=int(verts.shape[1]), L=int(verts.shape[0]))
    else:
        P[:, :, 1] -= P[:, :, 1].min()
        P[:, :, [0, 2]] -= P[0:1, 0:1, [0, 2]]
    resp["joints"] = P.tolist()
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
