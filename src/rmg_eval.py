"""
RMG conditional benchmark with the OFFICIAL Guo evaluators. Generates motion at each test caption's GT
length (CFG), converts to 263, scores FID / R-precision / MM-Dist / Diversity. Reuses the validated
motion_real harness (real-vs-real FID 0.001).

Exposes reusable functions (gather_test / prep_refs / eval_model) so the training-time eval monitor
(rmg_eval_monitor.py) can reuse the exact same scoring.

    python rmg_eval.py --ckpt runs/rmg_base/model.pth --n 1024 --guidance 6.5 --weights raw
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.float = float
import torch

from eval_official import (OfficialEvaluator, prep_motion, collect_motion_embeddings,
                           split_ids, HML, DIM_POSE, _eval_with_text_once)
from eval_cond import load_lines, eval_text_embeddings
import qwen_text
from rmg_to_263 import rmg_to_263
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity

UNIT, MAXLEN = 4, 196


def crop_len(T):
    return min((T // UNIT) * UNIT, MAXLEN)


def _div(emb, seed):
    np.random.seed(seed)
    return float(calculate_diversity(emb, 300 if len(emb) > 300 else len(emb) - 1))


def gather_test(n):
    """Up to n test (raw_caption, tokens, gt_len, real_263) tuples."""
    raws, toks, gtlens, real263 = [], [], [], []
    for mid in split_ids("test"):
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        if not os.path.exists(f):
            continue
        L = load_lines(mid)
        if not L:
            continue
        x = np.load(f).astype(np.float32)
        if x.ndim != 2 or x.shape[1] != DIM_POSE:
            continue
        T = crop_len(x.shape[0])
        if T < 40:
            continue
        raws.append(L[0][0]); toks.append(L[0][1]); gtlens.append(T); real263.append(x[:T])
        if len(raws) >= n:
            break
    return raws, toks, gtlens, real263


def prep_refs(ev, raws, toks, real263, seed=0):
    """Fixed-across-checkpoints references: real motion embeddings + Qwen cond + GloVe eval text."""
    rm, rl = [], []
    for x in real263:
        mp, ml = prep_motion(ev, x); rm.append(mp); rl.append(ml)
    real_emb = collect_motion_embeddings(ev, torch.from_numpy(np.stack(rm)), torch.tensor(rl), drop_last=False)
    mur, cr = calculate_activation_statistics(real_emb)
    return {"mur": mur, "cr": cr, "div_real": _div(real_emb, seed),
            "cond": qwen_text.encode(raws, device=ev.device),
            "eval_txt": eval_text_embeddings(ev, toks, ev.device)}


def eval_model(ev, model, flow, gtlens, refs, guidance=6.5, steps=100, gen_batch=32, seed=0):
    """Generate at GT lengths (CFG) -> 263 -> official metrics. Returns a metrics dict."""
    cond = refs["cond"]
    gen = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for s in range(0, len(gtlens), gen_batch):
            ce = cond[s:s + gen_batch].to(ev.device); bn = ce.shape[0]
            lens = gtlens[s:s + bn]; Lmax = max(lens)
            mask = torch.zeros(bn, Lmax, dtype=torch.bool, device=ev.device)
            for i, T in enumerate(lens):
                mask[i, :T] = True
            tr, q = flow.sample(model, bn, Lmax, mask=mask, text=ce, guidance=guidance, n_steps=steps, device=ev.device)
            tr, q = tr.cpu(), q.cpu()
            for i, T in enumerate(lens):
                gen.append(rmg_to_263(tr[i, :T], q[i, :T], off=None).numpy().astype(np.float32))
    gm, gl = [], []
    for dd in gen:
        mp, ml = prep_motion(ev, dd); gm.append(mp); gl.append(ml)
    gen_emb = collect_motion_embeddings(ev, torch.from_numpy(np.stack(gm)), torch.tensor(gl), drop_last=False)
    mug, cg = calculate_activation_statistics(gen_emb)
    fid = float(calculate_frechet_distance(refs["mur"], refs["cr"], mug, cg))
    order = np.random.RandomState(seed).permutation(len(gen_emb)).tolist()
    tk, mm, total = _eval_with_text_once(refs["eval_txt"][:len(gen_emb)], gen_emb, 3, order)
    R = tk / total
    return {"fid": fid, "div": _div(gen_emb, seed), "R_top1": float(R[0]), "R_top2": float(R[1]),
            "R_top3": float(R[2]), "mm_dist": float(mm / total)}


def load_model(ckpt, weights, dev):
    c = torch.load(ckpt, map_location=dev)
    model = RMGTransformer(**c["config"]).to(dev)
    key = "state_dict" if weights == "raw" else "ema_state_dict"
    model.load_state_dict(c.get(key, c["state_dict"])); model.eval()
    flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))
    return model, flow, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/rmg_base/model.pth")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--guidance", type=float, nargs="+", default=[6.5])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    ev = OfficialEvaluator(dev)
    model, flow, c = load_model(a.ckpt, a.weights, dev)
    raws, toks, gtlens, real263 = gather_test(a.n)
    refs = prep_refs(ev, raws, toks, real263, a.seed)
    print(f"[rmg-eval] {len(raws)} captions  step={c.get('step','?')}  weights={a.weights}  real-Div {refs['div_real']:.2f}")

    res = {"_meta": {"n": len(raws), "step": c.get("step", "?"), "weights": a.weights}, "div_real": refs["div_real"], "by_guidance": {}}
    for g in a.guidance:
        m = eval_model(ev, model, flow, gtlens, refs, guidance=g, steps=a.steps, gen_batch=a.gen_batch, seed=a.seed)
        res["by_guidance"][f"{g}"] = m
        print(f"g={g}: FID={m['fid']:.3f}  Div={m['div']:.3f}  R@1={m['R_top1']:.3f} R@3={m['R_top3']:.3f}  MMDist={m['mm_dist']:.3f}")
    json.dump(res, open(f"{a.out}/metrics_rmg.json", "w"), indent=2)
    print("wrote", f"{a.out}/metrics_rmg.json")


if __name__ == "__main__":
    main()
