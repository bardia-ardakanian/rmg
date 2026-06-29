"""
Train RMG-base on HumanML3D (faithful reproduction). Recipe = paper Table 7:
AdamW, max LR 1e-4, cosine schedule + warmup ratio 0.08, effective batch 256 (batch x grad_accum),
150k steps, grad clip 0.5, EMA, classifier-free dropout p=0.1.

Live monitoring: tqdm bar + runs/<out>/progress.txt + auto-updating loss_curve.png + **TensorBoard**
(scalars: loss / loss_ema / lr / grad_norm, and periodic generated-sample skeleton figures).

    python rmg_train.py --steps 150000 --batch 64 --accum 4 --out runs/rmg_base
    tensorboard --logdir runs --port 6006     # view via:  ssh -L 6006:localhost:6006 <gpu-host>
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
import s3
import rmg_data
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def cosine_warmup(opt, steps, warmup_frac=0.08, min_ratio=0.0):
    warmup = max(1, int(warmup_frac * steps))

    def fn(s):
        if s < warmup:
            return s / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def sample_figure(model, flow, dataset, dev, L=120, guidance=6.5):
    """Generate 2 conditioned samples and draw a 3-frame skeleton strip (for TensorBoard)."""
    from eval_hml import positions, _skel
    parents = _skel.parents()
    caps = dataset["cap_table"][:2].to(dev)
    tr, q = flow.sample(model, 2, L, text=caps, guidance=guidance, n_steps=50, device=dev)
    R = s3.quat_to_matrix(q.cpu())
    fig = plt.figure(figsize=(9, 3))
    for r in range(2):
        P = positions(tr[r].cpu(), R[r]).numpy()
        for c, fr in enumerate([0, L // 2, L - 1]):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            p = P[fr]
            for j, par in enumerate(parents):
                if par >= 0:
                    ax.plot([p[par, 0], p[j, 0]], [p[par, 2], p[j, 2]], [p[par, 1], p[j, 1]], c="#c0392b", lw=1.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 1.6))
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)            # effective batch = batch*accum (target 256)
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
    ap.add_argument("--max_motions", type=int, default=None)
    ap.add_argument("--cache", default="cache_rmg_train.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--sample_every", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    tb = SummaryWriter(os.path.join(a.out, "tb"))

    d = rmg_data.build_dataset("train", max_motions=a.max_motions, device=dev, cache=a.cache)
    N = d["trans"].shape[0]
    print(f"motions={N}  captions={d['cap_table'].shape[0]}  eff_batch={a.batch*a.accum}")

    model = RMGTransformer(dim=a.dim, num_layers=a.layers, num_heads=a.heads, ff_mult=a.ff_mult).to(dev)
    flow = RMGFlow(sigma_trans=a.sigma_trans, sigma_rot=a.sigma_rot)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = cosine_warmup(opt, a.steps, a.warmup_frac)
    ema = EMA(model, a.ema_decay)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  device={dev}")

    def save():
        torch.save({"state_dict": model.state_dict(), "ema_state_dict": ema.shadow,
                    "config": model.get_config(), "sigma_trans": a.sigma_trans, "sigma_rot": a.sigma_rot,
                    "step": s}, os.path.join(a.out, "model.pth"))

    hist = []
    loss_ema = None
    t0 = time.time()
    model.train()
    pbar = tqdm(range(a.steps), dynamic_ncols=True)
    for s in pbar:
        opt.zero_grad()
        tot = 0.0
        for _ in range(a.accum):
            idx = torch.randint(0, N, (a.batch,))
            bt, bq, text, mask = rmg_data.make_batch(d, idx, dev)
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
                f.write(f"[rmg_base] step {s+1}/{a.steps} ({100*(s+1)/a.steps:5.1f}%)\n[{bar}]\n"
                        f"loss {tot:.4f}  ema {loss_ema:.4f}  lr {lr:.2e}  elapsed {el:.0f}s  eta {eta:.0f}s\n")
        if s % 500 == 0 and hist:
            xs = [h[0] for h in hist]
            plt.figure(figsize=(7, 4)); plt.plot(xs, [h[1] for h in hist], alpha=.3, label="loss")
            plt.plot(xs, [h[2] for h in hist], label="ema"); plt.legend(); plt.xlabel("step"); plt.ylabel("loss")
            plt.tight_layout(); plt.savefig(os.path.join(a.out, "loss_curve.png"), dpi=100); plt.close()
        if s > 0 and s % a.sample_every == 0:
            try:
                fig = sample_figure(model, flow, d, dev); tb.add_figure("samples", fig, s); plt.close(fig)
            except Exception as e:
                print("sample viz skipped:", e)
            model.train()
        if s > 0 and s % a.save_every == 0:
            save()
    save()
    tb.close()
    print(f"saved -> {a.out}/model.pth")


if __name__ == "__main__":
    main()
