"""
HumanML3D -> RMG representation (R^3 translation + 22 S^3 quaternions), variable length + mask,
with cached Qwen3-Embedding text vectors.

Reuses motion_real/hml_data.feat_to_rep (263 -> trans + SO(3)); converts rotations to quaternions.
Motions are kept full-length (filtered to [40,196] frames, the official t2m range) and padded to 196;
training batches take a random coin2 crop (mult of unit_length=4) with a mask, matching the T2M loader.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import hml_data
import s3
import qwen_text

HML = hml_data.HML
MAXLEN = 196
MINLEN = 40
UNIT = 4


def load_captions(mid):
    out = []
    path = os.path.join(HML, "texts", mid + ".txt")
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.strip().split("#")
        if len(p) < 4:
            continue
        try:
            f, to = float(p[2]), float(p[3])
        except ValueError:
            continue
        if f == 0.0 and to == 0.0 and p[0].strip():
            out.append(p[0].strip())
    return out


def build_dataset(split="train", max_motions=None, device="cuda", cache=None):
    if cache and os.path.exists(cache):
        return torch.load(cache)
    trans_l, quat_l, len_l, all_caps, cap_ranges = [], [], [], [], []
    for mid in hml_data.split_ids(split):
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        if not os.path.exists(f):
            continue
        caps = load_captions(mid)
        if not caps:
            continue
        try:
            data = torch.from_numpy(np.load(f)).float()
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] != 263 or data.shape[0] < MINLEN:
            continue
        data = data[:MAXLEN]
        T = data.shape[0]
        trans, R = hml_data.feat_to_rep(data)                       # (T,3),(T,22,3,3)
        q = s3.canonical(s3.matrix_to_quat(R))                      # (T,22,4)
        tp = torch.zeros(MAXLEN, 3); tp[:T] = trans
        qp = torch.zeros(MAXLEN, 22, 4); qp[:, :, 0] = 1.0; qp[:T] = q   # pad with identity quat
        trans_l.append(tp); quat_l.append(qp); len_l.append(T)
        s = len(all_caps); all_caps.extend(caps); cap_ranges.append((s, s + len(caps)))
        if max_motions and len(len_l) >= max_motions:
            break
    if not len_l:
        raise RuntimeError("no motions built")
    cap_table = qwen_text.encode(all_caps, device=device)
    Kmax = max(e - s for s, e in cap_ranges)
    cap_pad = torch.full((len(cap_ranges), Kmax), -1, dtype=torch.long)
    cap_cnt = torch.zeros(len(cap_ranges), dtype=torch.long)
    for i, (s, e) in enumerate(cap_ranges):
        cap_pad[i, : e - s] = torch.arange(s, e); cap_cnt[i] = e - s
    d = dict(trans=torch.stack(trans_l), quats=torch.stack(quat_l), lengths=torch.tensor(len_l),
             cap_table=cap_table, cap_pad=cap_pad, cap_cnt=cap_cnt)
    if cache:
        torch.save(d, cache)
    return d


def make_batch(d, idx, device):
    """Random coin2 crop (mult of 4) per sample, xz-recentered to origin, padded to 196 + mask, + text."""
    B = len(idx)
    trans, quats, lengths = d["trans"][idx], d["quats"][idx], d["lengths"][idx]
    bt = torch.zeros(B, MAXLEN, 3)
    bq = torch.zeros(B, MAXLEN, 22, 4); bq[..., 0] = 1.0
    mask = torch.zeros(B, MAXLEN, dtype=torch.bool)
    for i in range(B):
        Li = int(lengths[i])
        m = (Li // UNIT) * UNIT
        if m > UNIT and random.random() < 1 / 3:                    # coin2 "double"
            m -= UNIT
        start = random.randint(0, Li - m) if Li > m else 0
        cr = trans[i, start:start + m].clone()
        cr[:, [0, 2]] -= cr[0, [0, 2]].clone()                     # recenter xz to origin (keep height)
        bt[i, :m] = cr
        bq[i, :m] = quats[i, start:start + m]
        mask[i, :m] = True
    cnt = d["cap_cnt"][idx]
    ks = (torch.rand(B) * cnt.float()).long().clamp_max_(cnt - 1)
    text = d["cap_table"][d["cap_pad"][idx, ks]]
    return bt.to(device), bq.to(device), text.to(device), mask.to(device)


if __name__ == "__main__":
    d = build_dataset(split="train", max_motions=200, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"motions={d['trans'].shape[0]}  captions={d['cap_table'].shape[0]}  "
          f"len[min/med/max]={int(d['lengths'].min())}/{int(d['lengths'].median())}/{int(d['lengths'].max())}")
    idx = torch.randperm(d["trans"].shape[0])[:8]
    bt, bq, text, mask = make_batch(d, idx, "cuda" if torch.cuda.is_available() else "cpu")
    print("batch:", tuple(bt.shape), tuple(bq.shape), tuple(text.shape), tuple(mask.shape))
    print("quat unit-norm err:", float((bq.norm(dim=-1) - 1).abs().max()),
          "| valid frames/sample:", mask.sum(1).tolist())
