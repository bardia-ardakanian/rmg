"""
Phase C conditional benchmark: text-conditioned generation evaluated with the OFFICIAL Guo evaluators.

For each TEST caption: generate a motion conditioned on its CLIP embedding (CFG), convert to 263, embed
with the official motion encoder. Then:
  - FID        : generated motion embeddings vs real test (T=64 windows)
  - R-precision: pair each generated motion with the SAME caption's GloVe-BiGRU text embedding
                 (the evaluator's encoder, NOT the CLIP we conditioned on -> no leakage), batches of 32
  - MM-Dist    : mean matched text<->motion distance
  - Diversity  : spread of generated motion embeddings
Swept over CFG guidance scales.

Caveat: our model is fixed T=64 (~3.2s); captions describe full clips, so long/sequential prompts are
only partially realizable. Numbers are on this evaluator checkpoint's scale (Real Top3=0.775, FID~0).

    python eval_cond.py --ckpt runs/hml_cond/model.pth --n 1024 --guidance 1 2.5 4 --out report
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
                           build_real_t64_windows, split_ids, HML, BATCH, _eval_with_text_once)
from hml_text import clip_encode
from rep_to_feat import rep_to_feat
from model import SpatioTemporalDiT
from flow import MotionFlow
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity

# published HumanML3D text-to-motion SOTA (paper-scale reference; our evaluator checkpoint reads ~0.97x
# on Real R-precision, so compare trends, not exact decimals)
SOTA = {
    "Real":      dict(fid=0.002, r3=0.797, mm=2.974, div=9.503),
    "RMG":       dict(fid=0.043, r3=0.797, mm=2.974, div=9.5),
    "MoMask":    dict(fid=0.045, r3=0.807, mm=2.958, div=9.6),
    "MLD":       dict(fid=0.473, r3=0.772, mm=3.196, div=9.724),
}


def load_lines(mid):
    """Full-clip (raw_caption, token_list) pairs from texts/<id>.txt."""
    out = []
    p = os.path.join(HML, "texts", mid + ".txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f, to = float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if f == 0.0 and to == 0.0 and parts[0].strip():
            toks = [t for t in parts[1].split(" ") if t]
            if toks:
                out.append((parts[0].strip(), toks))
    return out


@torch.no_grad()
def eval_text_embeddings(ev, toks, dev):
    """GloVe-BiGRU (official evaluator) text embeddings for R-precision, in input order."""
    out = np.zeros((len(toks), 512), dtype=np.float32)
    for s in range(0, len(toks), BATCH):
        chunk = toks[s:s + BATCH]
        we, po, sl = [], [], []
        for tk in chunk:
            w, p, l = ev.vectorize_caption(tk); we.append(w); po.append(p); sl.append(l)
        order = sorted(range(len(sl)), key=lambda i: sl[i], reverse=True)
        we_t = torch.from_numpy(np.stack([we[i] for i in order])).to(dev)
        po_t = torch.from_numpy(np.stack([po[i] for i in order])).to(dev)
        sl_t = torch.tensor([sl[i] for i in order])
        te = ev.text_enc(we_t, po_t, sl_t).cpu().numpy()
        for k, i in enumerate(order):
            out[s + i] = te[k]
    return out


@torch.no_grad()
def generate(model, flow, cond, T, ts, dev, steps, gen_batch, guidance, seed):
    reps = []
    torch.manual_seed(seed)
    for s in range(0, cond.shape[0], gen_batch):
        ce = cond[s:s + gen_batch].to(dev)
        gt, gR = flow.sample(model, ce.shape[0], T, n_steps=steps, device=dev, text_emb=ce, guidance=guidance)
        gt = (gt * ts).cpu(); gR = gR.cpu()
        for i in range(gt.shape[0]):
            reps.append(rep_to_feat(gt[i], gR[i], off=None).numpy().astype(np.float32))
    return reps


def embed_reps(ev, reps):
    mots, lens = [], []
    for d in reps:
        mp, ml = prep_motion(ev, d); mots.append(mp); lens.append(ml)
    return collect_motion_embeddings(ev, torch.from_numpy(np.stack(mots)),
                                     torch.tensor(lens, dtype=torch.long), drop_last=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/hml_cond/model.pth")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.5, 4.0])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gen_batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    ev = OfficialEvaluator(dev)
    c = torch.load(a.ckpt, map_location=dev)
    model = SpatioTemporalDiT(**c["config"]).to(dev)
    model.load_state_dict(c.get("ema_state_dict", c["state_dict"])); model.eval()
    flow = MotionFlow("vfm", c["sigma_rot"], c["sigma_trans"], c.get("w_trans", 1.0), amp=False)
    T, ts = c["T"], c["trans_std"]

    raws, toks = [], []
    for mid in split_ids("test"):
        if not os.path.exists(os.path.join(HML, "new_joint_vecs", mid + ".npy")):
            continue
        L = load_lines(mid)
        if not L:
            continue
        raws.append(L[0][0]); toks.append(L[0][1])
        if len(raws) >= a.n:
            break
    print(f"[cond-eval] {len(raws)} test captions")

    cond_all = clip_encode(raws, device=dev)                 # CLIP (for conditioning)
    eval_txt = eval_text_embeddings(ev, toks, dev)           # GloVe-BiGRU (for R-precision)

    rm, rl = build_real_t64_windows(ev, max(1024, len(raws)), seed=a.seed)
    real_emb = collect_motion_embeddings(ev, rm, rl, drop_last=False)
    mur, cr = calculate_activation_statistics(real_emb)
    np.random.seed(a.seed); div_real = float(calculate_diversity(real_emb, 300))

    res = {"_meta": {"n": len(raws), "steps": a.steps, "ckpt": a.ckpt, "real_div": div_real}, "by_guidance": {}}
    for g in a.guidance:
        reps = generate(model, flow, cond_all, T, ts, dev, a.steps, a.gen_batch, g, a.seed)
        gen_emb = embed_reps(ev, reps)
        mug, cg = calculate_activation_statistics(gen_emb)
        fid = float(calculate_frechet_distance(mur, cr, mug, cg))
        np.random.seed(a.seed); div = float(calculate_diversity(gen_emb, 300))
        order = np.random.RandomState(a.seed).permutation(len(gen_emb)).tolist()
        tk_cnt, mm, total = _eval_with_text_once(eval_txt[:len(gen_emb)], gen_emb, 3, order)
        R = (tk_cnt / total)
        res["by_guidance"][f"{g}"] = {"fid": fid, "diversity": div, "R_top1": float(R[0]),
                                      "R_top2": float(R[1]), "R_top3": float(R[2]), "mm_dist": float(mm / total),
                                      "n_pairs": total}
        print(f"g={g:>4}: FID={fid:7.3f}  Div={div:6.3f}  R@1={R[0]:.3f} R@2={R[1]:.3f} R@3={R[2]:.3f}  "
              f"MMDist={mm/total:.3f}")

    # markdown table
    lines = ["## Phase C — text-conditioned VFM (HumanML3D, official evaluator, T=64)", "",
             f"_{len(raws)} test captions; real-T64 Diversity={div_real:.3f}; evaluator checkpoint scale "
             "(Real Top3=0.775). Fixed T=64 caveat applies._", "",
             "| guidance | FID↓ | R@1↑ | R@2↑ | R@3↑ | MM-Dist↓ | Diversity |",
             "|---|---|---|---|---|---|---|"]
    for g, m in res["by_guidance"].items():
        lines.append(f"| {g} | {m['fid']:.3f} | {m['R_top1']:.3f} | {m['R_top2']:.3f} | {m['R_top3']:.3f} "
                     f"| {m['mm_dist']:.3f} | {m['diversity']:.3f} |")
    lines += ["", "Published reference (paper-scale; our evaluator reads ~0.97x on Real R-prec):", "",
              "| method | FID | R@3 | MM-Dist | Div |", "|---|---|---|---|---|"]
    for k, v in SOTA.items():
        lines.append(f"| {k} | {v['fid']} | {v['r3']} | {v['mm']} | {v['div']} |")
    md = "\n".join(lines)
    json.dump(res, open(f"{a.out}/metrics_cond.json", "w"), indent=2)
    open(f"{a.out}/metrics_cond.md", "w").write(md)
    print("\n" + md)
    print(f"\nwrote {a.out}/metrics_cond.json")


if __name__ == "__main__":
    main()
