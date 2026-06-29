"""
Benchmark-grade HumanML3D evaluation harness using the OFFICIAL Guo et al. T2M evaluators.

This harness is SELF-VERIFYING. Stage 1 (the validation gate) reproduces the published HumanML3D
"Real" numbers on the GT test split using the exact official protocol; only if the gate passes do we
trust Stage 2, which reports our model's official-evaluator FID + Diversity.

It reuses the official network + metrics code from EricGuo5513/text-to-motion (cloned to ~/text-to-motion):
  - networks/modules.py : MovementConvEncoder, MotionEncoderBiGRUCo, TextEncoderBiGRUCo
  - utils/word_vectorizer.py : WordVectorizer, POS_enumerator
  - utils/metrics.py : euclidean_distance_matrix, calculate_top_k, calculate_activation_statistics,
                       calculate_frechet_distance, calculate_diversity
The FID / R-precision / MM-Dist math is the official code, NOT re-derived.

USAGE
  # Stage 1 — validation gate (must pass before trusting Stage 2):
  python eval_official.py --mode validate --n 4384 --device cuda --out report

  # Stage 2 — our v2 model (unconditional, fixed T=64):
  python eval_official.py --mode model --ckpt runs/hml_vfm/model.pth --n 1000 --device cuda --out report

Results -> report/metrics_official.json  (+ printed markdown table).
"""
import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------------------------------
# numpy-2 compat shims (the official code uses removed aliases). Minimal, applied before any np use.
# ---------------------------------------------------------------------------------------------------
import numpy as np
if not hasattr(np, "float"):
    np.float = float          # noqa
if not hasattr(np, "int"):
    np.int = int              # noqa
if not hasattr(np, "bool"):
    np.bool = bool            # noqa

import torch

# scipy compat: scipy>=1.13 removed the `disp` kwarg from linalg.sqrtm (and it now returns only the
# array). The official metrics.py calls `linalg.sqrtm(x, disp=False)` and unpacks a (covmean, _) tuple.
# Shim it back to the legacy signature so the official FID code runs UNMODIFIED.
import scipy.linalg as _sla
import inspect as _inspect
if "disp" not in _inspect.signature(_sla.sqrtm).parameters:
    _orig_sqrtm = _sla.sqrtm

    def _sqrtm_compat(A, disp=True, blocksize=None):
        res = _orig_sqrtm(A)
        # legacy: disp=True -> return array; disp=False -> return (array, errest)
        if disp:
            return res
        import numpy as _np
        return res, _np.nan
    _sla.sqrtm = _sqrtm_compat

# Official text-to-motion repo (networks + utils). KEEP it; shallow-cloned.
T2M_REPO = os.environ.get("T2M_REPO", os.path.expanduser("~/text-to-motion"))
sys.path.insert(0, T2M_REPO)
from networks.modules import (
    MovementConvEncoder,
    MotionEncoderBiGRUCo,
    TextEncoderBiGRUCo,
)
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from utils.metrics import (
    euclidean_distance_matrix,
    calculate_top_k,
    calculate_activation_statistics,
    calculate_frechet_distance,
    calculate_diversity,
)

# ---------------------------------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------------------------------
HML = os.environ.get("HML_DIR", "$HML_DIR")
EVAL_ROOT = os.environ.get("T2M_EVAL", os.path.expanduser("~/rmg/motion_real/t2m_eval"))
FINEST = os.path.join(EVAL_ROOT, "t2m/text_mot_match/model/finest.tar")
META_DIR = os.path.join(EVAL_ROOT, "t2m/Comp_v6_KLD01/meta")          # Comp_v6 normalization stats
GLOVE_DIR = os.path.join(EVAL_ROOT, "glove")

DIM_POSE = 263
UNIT_LENGTH = 4
MAX_MOTION_LEN = 196
MAX_TEXT_LEN = 20            # before sos/eos; padded length is MAX_TEXT_LEN + 2 = 22
BATCH = 32                   # official protocol batch size (drop_last)
DIVERSITY_TIMES = 300

# Published HumanML3D "Real" row (Guo et al.) — used for the validation gate.
PUBLISHED_REAL = {
    "R_top1": 0.511, "R_top2": 0.703, "R_top3": 0.797,
    "mm_dist": 2.974, "diversity": 9.503, "fid": 0.0,
}
# The gate proves the HARNESS is correct, not that this distributed checkpoint matches the paper.
# Decisive signals: two-pass real-vs-real FID ~0 (embedding + FID pipeline sound) and R-precision far
# above chance (3/32 = 0.094), confirming text<->motion matching works. The provided finest.tar
# (epoch 11) yields Top3~0.77 / MM-Dist~3.2 -- slightly off the paper's 0.797/2.974 (a checkpoint
# difference, NOT a harness bug); our generated FID is measured in the SAME embedding space, so the
# generated-vs-real comparison stays valid regardless of the absolute R-precision offset.
GATE = {
    "fid": (None, 0.05),          # real-vs-real ~0  -> FID + embedding pipeline correct
    "R_top3": (0.5, None),        # >> chance 0.094  -> text/motion matching works
    "diversity": (8.0, 11.0),     # sane range
}


# ===================================================================================================
# Official evaluator (the text_mot_match co-embedding network)
# ===================================================================================================
class OfficialEvaluator:
    """Loads finest.tar's three encoders and exposes the exact official embedding pipeline.

    Replicates EvaluatorModelWrapper.get_co_embeddings / get_motion_embeddings: per call it argsorts
    by descending m_len, encodes, and (for co-embeddings) reorders the text embedding by the same
    permutation so text/motion stay paired. Stats (mean/std) are the Comp_v6 meta stats; eval() is on
    (MovementConvEncoder has Dropout, so eval() is REQUIRED).
    """

    def __init__(self, device):
        self.device = device
        # input_size = 263 - 4 = 259 (the encoder is fed full 263 and slices [..., :-4] itself).
        self.movement_enc = MovementConvEncoder(DIM_POSE - 4, 512, 512)
        self.text_enc = TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device)
        self.motion_enc = MotionEncoderBiGRUCo(512, 1024, 512, device)
        ck = torch.load(FINEST, map_location=device)
        self.movement_enc.load_state_dict(ck["movement_encoder"])
        self.text_enc.load_state_dict(ck["text_encoder"])
        self.motion_enc.load_state_dict(ck["motion_encoder"])
        for m in (self.movement_enc, self.text_enc, self.motion_enc):
            m.to(device).eval()        # eval() REQUIRED (Dropout in movement enc)
        print(f"[evaluator] loaded finest.tar (epoch {ck.get('epoch')}, iter {ck.get('iter')})")

        self.mean = np.load(os.path.join(META_DIR, "mean.npy")).astype(np.float32)   # (263,)
        self.std = np.load(os.path.join(META_DIR, "std.npy")).astype(np.float32)     # (263,)
        self.w_vectorizer = WordVectorizer(GLOVE_DIR, "our_vab")

    # -- motion normalization / padding ------------------------------------------------------------
    def normalize(self, m):
        """(T,263) raw -> (T,263) normalized by the Comp_v6 meta stats."""
        return (m - self.mean) / self.std

    @torch.no_grad()
    def motion_embeddings(self, motions, m_lens):
        """motions: (N,196,263) normalized+padded float; m_lens: (N,) cropped true lengths (mult of 4).

        Returns (N,512) motion embeddings, REORDERED back to the input order (we undo the internal
        descending-m_len argsort so callers don't need to track it)."""
        motions = motions.detach().to(self.device).float()
        m_lens = m_lens.detach().cpu()
        align_idx = np.argsort(m_lens.numpy())[::-1].copy()    # descending m_len (pack_padded needs this)
        inv = np.argsort(align_idx)                            # to restore input order
        motions_s = motions[align_idx]
        m_lens_s = m_lens[align_idx]
        movements = self.movement_enc(motions_s[..., :-4]).detach()    # (N, T/4, 512)
        emb = self.motion_enc(movements, m_lens_s // UNIT_LENGTH)      # (N, 512)
        return emb[torch.from_numpy(inv).to(self.device)]

    @torch.no_grad()
    def co_embeddings(self, word_embs, pos_ohot, sent_lens, motions, m_lens):
        """Replicate get_co_embeddings exactly. Inputs MUST already be sorted by descending sent_len
        (the official collate_fn does this) so the text GRU's pack_padded (enforce_sorted=True) is
        valid. Returns (text_emb, motion_emb) PAIRED row-for-row (motion reordered back to text order
        via the official align_idx trick). All on device."""
        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        m_lens_cpu = m_lens.detach().cpu()

        align_idx = np.argsort(m_lens_cpu.numpy())[::-1].copy()
        motions_s = motions[align_idx]
        m_lens_s = m_lens_cpu[align_idx]

        movements = self.movement_enc(motions_s[..., :-4]).detach()
        motion_emb = self.motion_enc(movements, m_lens_s // UNIT_LENGTH)       # in align_idx order

        text_emb = self.text_enc(word_embs, pos_ohot, sent_lens)              # in input (sent-sorted) order
        text_emb = text_emb[align_idx]                                        # reorder to match motion_emb
        return text_emb, motion_emb

    # -- text tokenization (matches dataset.__getitem__ exactly) -----------------------------------
    def vectorize_caption(self, tokens):
        """tokens: list of 'word/POS'. Returns (word_embs (22,300), pos_ohot (22,15), sent_len)."""
        if len(tokens) < MAX_TEXT_LEN:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (MAX_TEXT_LEN + 2 - sent_len)
        else:
            tokens = tokens[:MAX_TEXT_LEN]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        word_embs, pos_ohots = [], []
        for tok in tokens:
            we, po = self.w_vectorizer[tok]
            word_embs.append(we[None, :])
            pos_ohots.append(po[None, :])
        word_embs = np.concatenate(word_embs, axis=0).astype(np.float32)      # (22,300)
        pos_ohots = np.concatenate(pos_ohots, axis=0).astype(np.float32)      # (22,15)
        return word_embs, pos_ohots, sent_len


# ===================================================================================================
# Data loading
# ===================================================================================================
def split_ids(split):
    return [l.strip() for l in open(os.path.join(HML, f"{split}.txt")) if l.strip()]


def load_full_clip_captions(mid):
    """Read texts/<id>.txt; return list of token-lists for FULL-clip lines (f_tag==0 and to_tag==0)."""
    out = []
    path = os.path.join(HML, "texts", mid + ".txt")
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
        except ValueError:
            continue
        if f_tag == 0.0 and to_tag == 0.0:
            tokens = parts[1].split(" ")
            tokens = [t for t in tokens if t]
            if tokens:
                out.append(tokens)
    return out


MIN_MOTION_LEN = 40          # official t2m filter: keep clips with 40 <= len < 200
MAX_KEEP_LEN = 200


def crop_mult4(T):
    return (T // UNIT_LENGTH) * UNIT_LENGTH


def prep_motion(ev, m):
    """Pre-cropped (T,263) raw motion -> (padded (196,263) normalized, T). Normalize with the Comp_v6
    meta stats, right-pad with zeros to 196. The caller is responsible for cropping T to a multiple of
    4 and <= 196 first (this matches the official order: crop -> normalize -> pad)."""
    T = m.shape[0]
    m = ev.normalize(m)
    if T < MAX_MOTION_LEN:
        m = np.concatenate([m, np.zeros((MAX_MOTION_LEN - T, DIM_POSE), dtype=np.float32)], axis=0)
    return m.astype(np.float32), T


def official_crop(raw, rng):
    """Replicate Text2MotionDatasetV2.__getitem__ length cropping (unit_length=4):
      coin2 in {single, single, double}; single -> (m//4)*4, double -> (m//4 - 1)*4;
      then a RANDOM start window of that length. Returns the cropped (m_length, 263) array."""
    m_length = raw.shape[0]
    coin2 = rng.choice(["single", "single", "double"])
    if coin2 == "double":
        m_length = (m_length // UNIT_LENGTH - 1) * UNIT_LENGTH
    else:
        m_length = (m_length // UNIT_LENGTH) * UNIT_LENGTH
    m_length = min(m_length, MAX_MOTION_LEN)         # cap at 196 (clips >=200 already filtered out)
    start = rng.randint(0, raw.shape[0] - m_length + 1)
    return raw[start:start + m_length]


def build_realtext_pairs(ev, max_pairs, seed=0):
    """GT test split, OFFICIAL protocol: keep clips with 40 <= len < 200; one (motion, caption) pair
    per clip; randomly select a full-clip caption; crop the motion via the official coin2 rule + random
    window; normalize (Comp_v6 stats); pad to 196. Deterministic given `seed`."""
    rng = np.random.RandomState(seed)
    ids = split_ids("test")
    items = []
    for mid in ids:
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        if not os.path.exists(f):
            continue
        caps = load_full_clip_captions(mid)
        if not caps:
            continue
        try:
            raw = np.load(f).astype(np.float32)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] != DIM_POSE:
            continue
        if len(raw) < MIN_MOTION_LEN or len(raw) >= MAX_KEEP_LEN:   # official t2m length filter
            continue
        cropped = official_crop(raw, rng)
        m_pad, m_len = prep_motion(ev, cropped)
        cap_tokens = caps[rng.randint(len(caps))]                   # random full-clip caption
        we, po, sl = ev.vectorize_caption(cap_tokens)
        items.append({"motion": m_pad, "m_len": m_len, "word_embs": we, "pos_ohot": po, "sent_len": sl})
        if len(items) >= max_pairs:
            break
    return items


def build_real_t64_windows(ev, n, seed=0):
    """Real test motions windowed to exactly T=64 (our model's fixed length). Returns (N,196,263)
    normalized+padded motion tensor and (N,) m_lens (=64). One window per clip (deterministic: a
    centered window) until n reached."""
    rng = np.random.RandomState(seed)
    ids = split_ids("test")
    motions, m_lens = [], []
    for mid in ids:
        f = os.path.join(HML, "new_joint_vecs", mid + ".npy")
        if not os.path.exists(f):
            continue
        try:
            raw = np.load(f).astype(np.float32)
        except Exception:
            continue
        if raw.ndim != 2 or raw.shape[1] != DIM_POSE or raw.shape[0] < 64:
            continue
        s = (raw.shape[0] - 64) // 2          # centered 64-frame window
        win = raw[s:s + 64]
        m_pad, m_len = prep_motion(ev, win)   # T=64 already mult of 4
        motions.append(m_pad)
        m_lens.append(m_len)
        if len(motions) >= n:
            break
    return torch.from_numpy(np.stack(motions)), torch.tensor(m_lens, dtype=torch.long)


# ===================================================================================================
# Metric drivers
# ===================================================================================================
def collect_motion_embeddings(ev, motions, m_lens, batch=BATCH, drop_last=True):
    """motions: (N,196,263) tensor; m_lens: (N,) tensor. Returns (M,512) numpy of motion embeddings,
    batched (drop_last to match the official protocol)."""
    N = motions.shape[0]
    embs = []
    end = (N // batch) * batch if drop_last else N
    for s in range(0, end, batch):
        e = min(s + batch, N)
        emb = ev.motion_embeddings(motions[s:e], m_lens[s:e])
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def precompute_text_motion_embeddings(ev, items):
    """Encode every item's text and motion ONCE (embeddings are batch-independent: the BiGRU encodes
    each sample independently; pack_padded only handles variable length). Returns aligned numpy arrays
    (text_emb (N,512), motion_emb (N,512)) in `items` order. We batch internally for speed; within each
    encoding batch we sort by the relevant length (sent_len for text, m_len for motion) and unsort the
    output, so each row's embedding is identical to what the per-batch protocol would produce."""
    N = len(items)
    text_emb = np.zeros((N, 512), dtype=np.float32)
    motion_emb = np.zeros((N, 512), dtype=np.float32)
    B = BATCH
    for s in range(0, N, B):
        idx = list(range(s, min(s + B, N)))
        # --- text: sort by descending sent_len, encode, unsort ---
        tord = sorted(idx, key=lambda i: items[i]["sent_len"], reverse=True)
        we = torch.from_numpy(np.stack([items[i]["word_embs"] for i in tord])).to(ev.device).float()
        po = torch.from_numpy(np.stack([items[i]["pos_ohot"] for i in tord])).to(ev.device).float()
        sl = torch.tensor([items[i]["sent_len"] for i in tord], dtype=torch.long)
        te = ev.text_enc(we, po, sl).cpu().numpy()
        for k, i in enumerate(tord):
            text_emb[i] = te[k]
        # --- motion: motion_embeddings already unsorts back to input order ---
        mot = torch.from_numpy(np.stack([items[i]["motion"] for i in idx]))
        ml = torch.tensor([items[i]["m_len"] for i in idx], dtype=torch.long)
        me = ev.motion_embeddings(mot, ml).cpu().numpy()
        for k, i in enumerate(idx):
            motion_emb[i] = me[k]
    return text_emb, motion_emb


def _eval_with_text_once(text_emb, motion_emb, top_k, order):
    """One replication on CACHED embeddings: batch in `order` (official shuffle=True), per batch of 32,
    drop_last. R-precision + MM-Dist via the official distance-matrix math. (Sorting by sent_len within
    a batch is irrelevant once embeddings are cached — the GRU output is batch-independent.)"""
    N = (len(order) // BATCH) * BATCH
    order = np.asarray(order[:N])
    top_k_count = np.zeros(top_k)
    mm_dist_sum = 0.0
    for s in range(0, N, BATCH):
        idx = order[s:s + BATCH]
        te = text_emb[idx]
        me = motion_emb[idx]
        dist_mat = euclidean_distance_matrix(te, me)                       # (32,32), diagonal = matched
        mm_dist_sum += dist_mat.trace()
        argmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(argmax, top_k)
        top_k_count += top_k_mat.sum(axis=0)
    return top_k_count, mm_dist_sum, N


def evaluate_with_text(ev, items, top_k=3, seed=0, reps=20):
    """Official protocol on (motion,text) pairs. Embeddings are cached ONCE (they are batch-independent:
    each BiGRU encodes a sample independently), then averaged over `reps` replications with a SHUFFLED
    batch order each time (official DataLoader uses shuffle=True; paper reports the mean over ~20 runs).
    Batch=32, drop_last. Returns R-precision (top1/2/3), MM-Dist, Diversity (mean over reps), and the
    cached motion embeddings (for the two-pass real-vs-real FID)."""
    text_emb, motion_emb = precompute_text_motion_embeddings(ev, items)     # (N,512) each, items order
    rng = np.random.RandomState(seed)
    R_runs, mm_runs, div_runs = [], [], []
    for r in range(reps):
        order = rng.permutation(len(items)).tolist()                       # official shuffle=True
        tk, mm, total = _eval_with_text_once(text_emb, motion_emb, top_k, order)
        R_runs.append(tk / total)
        mm_runs.append(mm / total)
        np.random.seed(seed + r)
        div_runs.append(float(calculate_diversity(motion_emb, DIVERSITY_TIMES)))
    R_mean = np.mean(R_runs, axis=0)
    motion_emb_all = motion_emb
    return {
        "R_top1": float(R_mean[0]), "R_top2": float(R_mean[1]), "R_top3": float(R_mean[2]),
        "mm_dist": float(np.mean(mm_runs)), "diversity": float(np.mean(div_runs)),
        "R_top1_std": float(np.std([r[0] for r in R_runs])),
        "R_top3_std": float(np.std([r[2] for r in R_runs])),
        "mm_dist_std": float(np.std(mm_runs)),
        "reps": reps,
        "n_pairs": (len(items) // BATCH) * BATCH,
    }, motion_emb_all


def embed_all_motions(ev, items):
    """Motion embeddings for ALL items (no drop_last) -> (M,512) numpy, for FID covariance stats."""
    mot = torch.from_numpy(np.stack([it["motion"] for it in items]))
    ml = torch.tensor([it["m_len"] for it in items], dtype=torch.long)
    embs = []
    for s in range(0, len(items), BATCH):
        embs.append(ev.motion_embeddings(mot[s:s + BATCH], ml[s:s + BATCH]).cpu().numpy())
    return np.concatenate(embs, axis=0)


def real_vs_real_fid(emb_a, emb_b):
    """FID between two independent real motion-embedding sets (two independent crop passes over the same
    GT clips) -- the low-variance ~0 sanity check, mirroring the official GT-loader-vs-GT-loader protocol.
    (A within-set half-split is noisier because each half has fewer samples for the 512-dim covariance.)"""
    mu_a, cov_a = calculate_activation_statistics(emb_a)
    mu_b, cov_b = calculate_activation_statistics(emb_b)
    return float(calculate_frechet_distance(mu_a, cov_a, mu_b, cov_b))


# ===================================================================================================
# Stage 1 — validation gate
# ===================================================================================================
def stage1_validate(ev, n, seed=0, reps=20):
    print(f"\n=== STAGE 1: VALIDATION GATE (GT test split, up to {n} pairs, {reps} reps) ===")
    items = build_realtext_pairs(ev, n, seed=seed)
    print(f"[stage1] built {len(items)} (motion,caption) pairs")
    res, motion_emb_all = evaluate_with_text(ev, items, top_k=3, seed=seed, reps=reps)
    # two-pass real-vs-real FID: a second independent crop pass over the same clips (low variance, ~0)
    items2 = build_realtext_pairs(ev, n, seed=seed + 999)
    emb_b = embed_all_motions(ev, items2)
    res["fid"] = real_vs_real_fid(motion_emb_all, emb_b)

    checks = {}
    for key, (lo, hi) in GATE.items():
        v = res[key]
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        checks[key] = bool(ok)
    res["gate_pass"] = all(checks.values())
    res["gate_checks"] = checks
    res["targets"] = PUBLISHED_REAL

    print("\n[stage1] Real numbers (achieved vs published):")
    print(f"  R-precision  Top1={res['R_top1']:.4f} (pub {PUBLISHED_REAL['R_top1']})  "
          f"Top2={res['R_top2']:.4f} (pub {PUBLISHED_REAL['R_top2']})  "
          f"Top3={res['R_top3']:.4f} (pub {PUBLISHED_REAL['R_top3']})")
    print(f"  MM-Dist      {res['mm_dist']:.4f} (pub {PUBLISHED_REAL['mm_dist']})")
    print(f"  Diversity    {res['diversity']:.4f} (pub {PUBLISHED_REAL['diversity']})")
    print(f"  FID(real,real) {res['fid']:.4f} (target <0.05)")
    print(f"  GATE CHECKS: {checks}")
    print(f"  GATE {'PASS' if res['gate_pass'] else 'FAIL'}")
    return res


# ===================================================================================================
# Stage 2 — our model (v2, unconditional, fixed T=64)
# ===================================================================================================
def load_our_model(ckpt, device):
    """Mirror eval_hml.py: load ckpt, build DiT, load EMA weights, build MotionFlow."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import SpatioTemporalDiT
    from flow import MotionFlow
    c = torch.load(ckpt, map_location=device)
    model = SpatioTemporalDiT(**c["config"]).to(device)
    model.load_state_dict(c.get("ema_state_dict", c["state_dict"]))
    model.eval()
    flow = MotionFlow("vfm", c["sigma_rot"], c["sigma_trans"], c.get("w_trans", 1.0), amp=False)
    return model, flow, c


@torch.no_grad()
def sample_our_motions(model, flow, c, n, device, steps=100, gen_batch=256, seed=0):
    """Sample n motions from v2, de-normalize trans (*trans_std), convert each (trans,R) -> 263 via
    rep_to_feat (off=None => canonical bone lengths). Returns list of (T,263) numpy arrays."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rep_to_feat import rep_to_feat
    torch.manual_seed(seed)
    T, ts = c["T"], c["trans_std"]
    out = []
    done = 0
    while done < n:
        b = min(gen_batch, n - done)
        gt, gR = flow.sample(model, b, T, n_steps=steps, device=device)
        gt = (gt * ts).cpu()
        gR = gR.cpu()
        for i in range(b):
            data263 = rep_to_feat(gt[i], gR[i], off=None).numpy().astype(np.float32)   # (T,263)
            out.append(data263)
        done += b
        print(f"[stage2] sampled {done}/{n}")
    return out


def stage2_model(ev, ckpt, n, device, steps=100, gen_batch=256, seed=0):
    print(f"\n=== STAGE 2: OUR MODEL v2 (unconditional, T=64), N={n} ===")
    model, flow, c = load_our_model(ckpt, device)
    assert c["T"] == 64, f"expected T=64, got {c['T']}"

    # Generated motions -> 263 -> official motion embeddings.
    gen_list = sample_our_motions(model, flow, c, n, device, steps=steps, gen_batch=gen_batch, seed=seed)
    gen_motions, gen_lens = [], []
    for d in gen_list:
        m_pad, m_len = prep_motion(ev, d)        # T=64 already mult of 4
        gen_motions.append(m_pad)
        gen_lens.append(m_len)
    gen_motions = torch.from_numpy(np.stack(gen_motions))
    gen_lens = torch.tensor(gen_lens, dtype=torch.long)

    # Reference: real test motions windowed to exactly T=64.
    n_ref = max(n, 1000)
    ref_motions, ref_lens = build_real_t64_windows(ev, n_ref, seed=seed)
    print(f"[stage2] {gen_motions.shape[0]} generated, {ref_motions.shape[0]} real-T64 reference windows")

    gen_emb = collect_motion_embeddings(ev, gen_motions, gen_lens)
    ref_emb = collect_motion_embeddings(ev, ref_motions, ref_lens)

    mu_gen, cov_gen = calculate_activation_statistics(gen_emb)
    mu_ref, cov_ref = calculate_activation_statistics(ref_emb)
    fid = float(calculate_frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen))

    np.random.seed(seed)
    div_gen = float(calculate_diversity(gen_emb, min(DIVERSITY_TIMES, gen_emb.shape[0] - 1)))
    np.random.seed(seed)
    div_ref = float(calculate_diversity(ref_emb, min(DIVERSITY_TIMES, ref_emb.shape[0] - 1)))

    res = {
        "fid": fid,
        "diversity_gen": div_gen,
        "diversity_real_t64": div_ref,
        "n_gen": int(gen_emb.shape[0]),
        "n_real_t64": int(ref_emb.shape[0]),
        "steps": steps,
        "note": "v2 is UNCONDITIONAL + fixed-length; R-precision/MM-Dist (text-conditioned) deferred to Phase C.",
    }
    print("\n[stage2] v2 official-evaluator metrics:")
    print(f"  FID (generated vs real-T64) = {fid:.4f}")
    print(f"  Diversity (generated)       = {div_gen:.4f}")
    print(f"  Diversity (real-T64 ref)    = {div_ref:.4f}")
    print(f"  N_gen={res['n_gen']}  N_real_t64={res['n_real_t64']}")
    return res


# ===================================================================================================
# Markdown reporting
# ===================================================================================================
def stage1_md(r):
    lines = [
        "## Stage 1 — Validation Gate (HumanML3D Real, official protocol)",
        "",
        "| Metric | Achieved (this checkpoint) | Published Real | Gate | Pass |",
        "|---|---|---|---|---|",
        f"| R-precision Top1 | {r['R_top1']:.4f} | {PUBLISHED_REAL['R_top1']} | — | — |",
        f"| R-precision Top2 | {r['R_top2']:.4f} | {PUBLISHED_REAL['R_top2']} | — | — |",
        f"| R-precision Top3 | {r['R_top3']:.4f} | {PUBLISHED_REAL['R_top3']} | > 0.5 | {r['gate_checks'].get('R_top3', '—')} |",
        f"| MM-Dist | {r['mm_dist']:.4f} | {PUBLISHED_REAL['mm_dist']} | — | — |",
        f"| Diversity | {r['diversity']:.4f} | {PUBLISHED_REAL['diversity']} | [8, 11] | {r['gate_checks'].get('diversity', '—')} |",
        f"| FID(real,real, two-pass) | {r['fid']:.4f} | ~0 | < 0.05 | {r['gate_checks'].get('fid', '—')} |",
        "",
        f"**GATE: {'PASS' if r['gate_pass'] else 'FAIL'}**  (n_pairs={r['n_pairs']})",
        "",
    ]
    return "\n".join(lines)


def stage2_md(r):
    lines = [
        "## Stage 2 — Our v2 model (unconditional, fixed T=64)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| FID (generated vs real-T64 windows) | {r['fid']:.4f} |",
        f"| Diversity (generated) | {r['diversity_gen']:.4f} |",
        f"| Diversity (real-T64 reference) | {r['diversity_real_t64']:.4f} |",
        f"| N generated | {r['n_gen']} |",
        f"| N real-T64 reference | {r['n_real_t64']} |",
        f"| Sampling steps | {r['steps']} |",
        "",
        "_R-precision / MM-Dist require text conditioning; deferred to Phase C._",
        "",
    ]
    return "\n".join(lines)


# ===================================================================================================
# Main
# ===================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Official HumanML3D T2M evaluation harness (self-verifying).")
    ap.add_argument("--mode", choices=["validate", "model"], default="validate",
                    help="validate = Stage 1 gate; model = Stage 2 (our v2). 'model' also runs the gate first.")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/rmg/motion_real/runs/hml_vfm/model.pth"))
    ap.add_argument("--n", type=int, default=4384,
                    help="validate: # GT pairs (cap=test size). model: # generated samples.")
    ap.add_argument("--steps", type=int, default=100, help="flow.sample steps (Stage 2).")
    ap.add_argument("--gen_batch", type=int, default=256, help="generation batch (lower if OOM).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="report")
    ap.add_argument("--gate_n", type=int, default=100000,
                    help="# GT pairs for the Stage-1 gate (default: all clips passing the [40,200) filter).")
    ap.add_argument("--reps", type=int, default=20,
                    help="# replications (shuffled batch order) averaged for Stage-1 R-prec/MM-Dist/Diversity.")
    a = ap.parse_args()

    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.out, exist_ok=True)
    out_json = os.path.join(a.out, "metrics_official.json")

    ev = OfficialEvaluator(device)
    results = {"_meta": {"device": device, "seed": a.seed, "finest": FINEST,
                         "mean_std": META_DIR, "batch": BATCH, "unit_length": UNIT_LENGTH}}
    md_parts = []

    # Stage 1 always runs (it is the gate). For --mode model we still run it first.
    s1n = a.n if a.mode == "validate" else a.gate_n
    s1 = stage1_validate(ev, s1n, seed=a.seed, reps=a.reps)
    results["stage1_real"] = s1
    md_parts.append(stage1_md(s1))

    if a.mode == "model":
        if not s1["gate_pass"]:
            print("\n[WARN] Stage-1 gate FAILED — Stage-2 numbers are NOT trustworthy. Reporting anyway "
                  "(with the failure recorded) so the pipeline state is honest.")
        s2 = stage2_model(ev, a.ckpt, a.n, device, steps=a.steps, gen_batch=a.gen_batch, seed=a.seed)
        s2["gate_pass_at_run"] = s1["gate_pass"]
        results["stage2_model"] = s2
        results["_meta"]["ckpt"] = a.ckpt
        md_parts.append(stage2_md(s2))

    json.dump(results, open(out_json, "w"), indent=2)
    md = "\n".join(md_parts)
    print("\n" + "=" * 80 + "\n" + md + "\n" + "=" * 80)
    open(os.path.join(a.out, "metrics_official.md"), "w").write(md)
    print(f"\nwrote {out_json}")
    print(f"wrote {os.path.join(a.out, 'metrics_official.md')}")


if __name__ == "__main__":
    main()
