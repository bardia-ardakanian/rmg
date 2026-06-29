"""
MotionMillion (rpr272, 30fps) -> RMG representation (R^3 translation + 22 S^3 quaternions).

The 272-dim "local rotation" layout (22 joints), ported from MotionMillion's recover_from_local_rotation:
  [0:2]      root linear velocity (x, z) in the heading-free frame
  [2:8]      global heading rotation increment (6D, relative per frame)
  [8:74]     local joint positions, heading removed (22 * 3); [9] = root height
  [74:140]   local joint velocities (22 * 3)
  [140:272]  joint rotations (22 * 6D continuous, Zhou et al.)

We rebuild the same thing RMG trains on: root global translation (integrate the heading-rotated root
velocity, height from the position block) and 22 joint rotations as quaternions (joint 0 = global root
orientation after applying the accumulated heading; joints 1..21 = local rotations straight from the 6D).
This is cleaner than HumanML3D: the rpr272 rotations are already proper SMPL-X-style local rotations.

On disk the released subset lives at  <MM_ROOT>/folder{0..9}/<id>.npy  (the split's source prefix is
dropped); split ids that the authors did not release are filtered out. Captions: <MM_ROOT>/texts/<name>.txt
(one caption per line) -- precomputed to Qwen embeddings by mm_prep.py.
"""
import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import s3

MM_ROOT = os.environ.get("MM_ROOT", "data/MotionMillion")
# split lists live inside split.tar.gz; mm_prep extracts them here (the dataset dir isn't ours to write)
MM_META = os.environ.get("MM_META", "cache/mm_meta")
NJ = 22
UNIT = 4
MINLEN = 60
MAXLEN = 200          # MotionMillion's t2m loader drops clips longer than this
SPLIT_REL = "split/version1/t2m_60_300"


# ----------------------------------------------------------------------------- rpr272 -> (trans, quats)
def _rot6d_to_mat(d):
    """(...,6) -> (...,3,3) via Gram-Schmidt, rows stacked (Zhou et al., matches MotionMillion)."""
    a1, a2 = d[..., :3], d[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def _accumulate(rel):
    """(T,3,3) relative rotations -> (T,3,3) cumulative, R[t] = rel[t] @ R[t-1] (MotionMillion order)."""
    out = [rel[0]]
    for t in range(1, rel.shape[0]):
        out.append(rel[t] @ out[-1])
    return torch.stack(out, 0)


def rpr272_to_rep(x):
    """x (T,272) float -> trans (T,3), quats (T,22,4). Faithful port of recover_from_local_rotation."""
    T = x.shape[0]
    rot = _rot6d_to_mat(x[:, 8 + 6 * NJ:8 + 12 * NJ].reshape(T, NJ, 6)).clone()   # (T,22,3,3)
    heading = _rot6d_to_mat(x[:, 2:8])                                            # (T,3,3) per-frame increment
    gh = _accumulate(heading)
    inv_gh = gh.transpose(-1, -2)
    rot[:, 0] = inv_gh @ rot[:, 0]                                                # global root orientation

    v = torch.zeros(T, 3)
    v[:, 0] = x[:, 0]
    v[:, 2] = x[:, 1]
    v[1:] = (inv_gh[:-1] @ v[1:].unsqueeze(-1)).squeeze(-1)                       # heading -> world
    trans = torch.cumsum(v, dim=0)
    trans[:, 1] = x[:, 9]                                                         # root height

    quats = s3.canonical(s3.matrix_to_quat(rot))
    return trans.float(), quats.float()


# ----------------------------------------------------------------------------- split / file resolution
def _disk_set():
    s = set()
    for d in sorted(glob.glob(os.path.join(MM_ROOT, "folder*"))):
        fk = os.path.basename(d)
        for fn in os.listdir(d):
            if fn.endswith(".npy"):
                s.add(f"{fk}/{fn[:-4]}")
    return s


def _folder_id(name):
    p = name.split("/")
    return f"{p[-2]}/{p[-1]}"


def read_split(split):
    for base in (MM_META, MM_ROOT):
        p = os.path.join(base, SPLIT_REL, split + ".txt")
        if os.path.exists(p):
            with open(p) as f:
                return [l.strip() for l in f if l.strip()]
    raise FileNotFoundError(
        f"{split}.txt not found under {MM_META} or {MM_ROOT}; run: "
        f"tar -xzf {MM_ROOT}/split.tar.gz -C {MM_META}")


def resolvable_clips(split, max_clips=None, shuffle_seed=0):
    """Return [(name, motion_path)] for split ids whose motion file exists on disk."""
    disk = _disk_set()
    ids = [n for n in read_split(split) if _folder_id(n) in disk]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(ids)
    if max_clips:
        ids = ids[:max_clips]
    return [(n, os.path.join(MM_ROOT, _folder_id(n) + ".npy")) for n in ids]


def caption_path(name):
    return os.path.join(MM_ROOT, "texts", name + ".txt")


# ----------------------------------------------------------------------------- dataset / loader
class MMDataset(Dataset):
    """Lazy: load one rpr272 clip, decode to (trans, quats), random coin2 crop, attach a text embedding."""
    def __init__(self, paths, caprows, emb, minlen=MINLEN, maxlen=MAXLEN, unit=UNIT):
        self.paths = paths
        self.caprows = caprows            # list[list[int]] -> rows into emb
        self.emb = emb                    # (M, 1024) float array (np memmap ok)
        self.minlen, self.maxlen, self.unit = minlen, maxlen, unit

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            x = np.load(self.paths[i])
        except Exception:
            return self[(i + 1) % len(self.paths)]
        T = x.shape[0]
        if T < self.minlen:
            return self[(i + 1) % len(self.paths)]
        T = min(T, self.maxlen)
        x = x[:T]
        m = (T // self.unit) * self.unit
        if m > self.unit and random.random() < 1.0 / 3.0:        # coin2 "double"
            m -= self.unit
        start = random.randint(0, T - m) if T > m else 0
        tr, q = rpr272_to_rep(torch.from_numpy(x[start:start + m]).float())
        tr[:, [0, 2]] -= tr[0, [0, 2]].clone()                   # recenter xz to origin, keep height
        rows = self.caprows[i]
        emb = torch.from_numpy(np.array(self.emb[random.choice(rows)], dtype=np.float32))   # writable copy
        return tr, q, emb, m


def collate_pad(batch):
    """Pad each batch to its own max length (dynamic) + build the frame mask."""
    L = max(b[3] for b in batch)
    B = len(batch)
    bt = torch.zeros(B, L, 3)
    bq = torch.zeros(B, L, NJ, 4); bq[..., 0] = 1.0
    text = torch.stack([b[2] for b in batch])
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, (tr, q, _, m) in enumerate(batch):
        bt[i, :m] = tr; bq[i, :m] = q; mask[i, :m] = True
    return bt, bq, text, mask


def make_loader(index_path, emb_path, batch=64, workers=8, minlen=MINLEN, maxlen=MAXLEN):
    """Build a DataLoader from the mm_prep index (paths + ragged caption rows) and the embedding memmap."""
    idx = np.load(index_path, allow_pickle=True)
    paths = list(idx["paths"])
    off = idx["capoff"]
    flat = idx["caprows"]
    caprows = [flat[off[i]:off[i + 1]].tolist() for i in range(len(paths))]
    emb = np.load(emb_path, mmap_mode="r")
    ds = MMDataset(paths, caprows, emb, minlen=minlen, maxlen=maxlen)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
        collate_fn=collate_pad, pin_memory=True, persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None)
    return loader, len(paths), emb.shape[0]


if __name__ == "__main__":
    # quick self-test: decode one clip, check quats are unit norm and rotations valid
    clips = resolvable_clips("train", max_clips=3)
    for name, path in clips:
        x = torch.from_numpy(np.load(path)).float()
        tr, q = rpr272_to_rep(x)
        R = s3.quat_to_matrix(q)
        det = torch.linalg.det(R)
        print(f"{name}: T={x.shape[0]:3d} trans_y[{tr[:,1].min():.2f},{tr[:,1].max():.2f}] "
              f"quat_norm_err={float((q.norm(dim=-1)-1).abs().max()):.2e} det[{det.min():.3f},{det.max():.3f}]")
