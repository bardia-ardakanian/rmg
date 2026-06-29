"""
Train RMG on MotionMillion (rpr272, ~458k released clips, 30fps). Same model/flow/recipe as the
HumanML3D reproduction, just a lazy DataLoader over the big set with precomputed Qwen text embeddings.

Run mm_prep.py first to build the index + embedding memmap, then:

    python rmg_train_mm.py --index cache/mm_train_index.npz --emb cache/mm_train_emb.npy \
        --steps 300000 --batch 64 --accum 4 --workers 12 --out runs/rmg_mm
    tensorboard --logdir runs --port 6006     # ssh -L 6006:localhost:6006 <host>

Live: tqdm bar + runs/<out>/progress.txt + loss_curve.png + TensorBoard (loss / loss_ema / lr / grad_norm).
"""
import argparse
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_data
from rmg_train import EMA, cosine_warmup
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff_mult", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.08)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--ema_decay", type=float, default=0.9999)
    ap.add_argument("--p_drop", type=float, default=0.1)
    ap.add_argument("--sigma_trans", type=float, default=1.0)
    ap.add_argument("--sigma_rot", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tb = SummaryWriter(os.path.join(a.out, "tb"))

    loader, n_clips, n_caps = mm_data.make_loader(a.index, a.emb, batch=a.batch, workers=a.workers)
    print(f"MotionMillion: clips={n_clips}  captions={n_caps}  eff_batch={a.batch*a.accum}  device={dev}")
    it = cycle(loader)

    model = RMGTransformer(dim=a.dim, num_layers=a.layers, num_heads=a.heads, ff_mult=a.ff_mult).to(dev)
    flow = RMGFlow(sigma_trans=a.sigma_trans, sigma_rot=a.sigma_rot)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = cosine_warmup(opt, a.steps, a.warmup_frac)
    ema = EMA(model, a.ema_decay)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    def save():
        torch.save({"state_dict": model.state_dict(), "ema_state_dict": ema.shadow,
                    "config": model.get_config(), "sigma_trans": a.sigma_trans, "sigma_rot": a.sigma_rot,
                    "step": s, "dataset": "motionmillion"}, os.path.join(a.out, "model.pth"))

    hist, loss_ema, t0 = [], None, time.time()
    model.train()
    pbar = tqdm(range(a.steps), dynamic_ncols=True)
    for s in pbar:
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.accum):
            bt, bq, text, mask = next(it)
            bt, bq, text, mask = bt.to(dev), bq.to(dev), text.to(dev), mask.to(dev)
            loss = flow.training_loss(model, bt, bq, text=text, mask=mask, p_drop=a.p_drop) / a.accum
            if torch.isfinite(loss):
                loss.backward(); tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step(); ema.update(model)
        loss_ema = tot if loss_ema is None else 0.98 * loss_ema + 0.02 * tot
        if s % a.log_every == 0:
            lr = sched.get_last_lr()[0]
            tb.add_scalar("train/loss", tot, s); tb.add_scalar("train/loss_ema", loss_ema, s)
            tb.add_scalar("train/lr", lr, s); tb.add_scalar("train/grad_norm", float(gn), s)
            pbar.set_description(f"loss {tot:.4f} ema {loss_ema:.4f} lr {lr:.1e}")
            hist.append((s, tot, loss_ema))
            el = time.time() - t0; eta = el / (s + 1) * (a.steps - s - 1)
            bar = ("#" * int(30 * (s + 1) / a.steps)).ljust(30)
            with open(os.path.join(a.out, "progress.txt"), "w") as f:
                f.write(f"[rmg_mm] step {s+1}/{a.steps} ({100*(s+1)/a.steps:5.1f}%)\n[{bar}]\n"
                        f"loss {tot:.4f}  ema {loss_ema:.4f}  lr {lr:.2e}  elapsed {el:.0f}s  eta {eta:.0f}s\n")
        if s % 500 == 0 and hist:
            xs = [h[0] for h in hist]
            plt.figure(figsize=(7, 4)); plt.plot(xs, [h[1] for h in hist], alpha=.3, label="loss")
            plt.plot(xs, [h[2] for h in hist], label="ema"); plt.legend(); plt.xlabel("step"); plt.ylabel("loss")
            plt.tight_layout(); plt.savefig(os.path.join(a.out, "loss_curve.png"), dpi=100); plt.close()
        if s > 0 and s % a.save_every == 0:
            save()
    save()
    tb.close()
    print(f"saved -> {a.out}/model.pth")


if __name__ == "__main__":
    main()
