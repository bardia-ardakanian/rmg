"""
Training-time eval monitor: watches runs/<out>/model.pth and, whenever the step advances enough,
runs a quick official eval (RAW weights — EMA is stale early) and logs FID / R-precision / Diversity /
MM-Dist to TensorBoard under eval/*, so quality curves appear alongside the training loss.

Runs in its own process (tmux), shares the GPU with training. No restart of training needed.

    python rmg_eval_monitor.py --run runs/rmg_base --n 256 --guidance 6.5 --every_steps 8000
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.tensorboard import SummaryWriter

from eval_official import OfficialEvaluator
import rmg_eval


def read_step(run):
    """Current step from progress.txt (cheap), -1 if unavailable."""
    p = os.path.join(run, "progress.txt")
    if not os.path.exists(p):
        return -1
    m = re.search(r"step (\d+)", open(p).read())
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/rmg_base")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--guidance", type=float, default=6.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--weights", choices=["ema", "raw"], default="raw")
    ap.add_argument("--every_steps", type=int, default=8000)
    ap.add_argument("--poll", type=int, default=120)            # seconds between progress checks
    ap.add_argument("--max_step", type=int, default=150000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(a.run, "model.pth")

    ev = OfficialEvaluator(dev)
    raws, toks, gtlens, real263 = rmg_eval.gather_test(a.n)
    refs = rmg_eval.prep_refs(ev, raws, toks, real263, seed=0)
    tb = SummaryWriter(os.path.join(a.run, "tb_eval"))         # same --logdir runs picks this up
    print(f"[evalmon] watching {ckpt}; {len(raws)} captions; real-Div {refs['div_real']:.2f}; "
          f"weights={a.weights}; every ~{a.every_steps} steps", flush=True)
    tb.add_scalar("eval/diversity_real", refs["div_real"], 0)

    last_eval = -a.every_steps
    stale = 0
    while True:
        step = read_step(a.run)
        if step < 0 or not os.path.exists(ckpt):
            time.sleep(a.poll); continue
        if step - last_eval < a.every_steps and step < a.max_step:
            stale = stale + 1 if step == read_step(a.run) else 0
            time.sleep(a.poll)
            # exit if training process is gone and we've already evaluated the latest
            if not os.popen("pgrep -f rmg_train.py").read().strip() and step <= last_eval:
                break
            continue
        try:
            # snapshot the checkpoint to avoid reading mid-save
            snap = os.path.join(a.run, "model_evalmon.pth")
            os.system(f"cp {ckpt} {snap}")
            model, flow, c = rmg_eval.load_model(snap, a.weights, dev)
            cstep = c.get("step", step)
            m = rmg_eval.eval_model(ev, model, flow, gtlens, refs, guidance=a.guidance,
                                    steps=a.steps, gen_batch=a.gen_batch, seed=0)
            for k in ("fid", "div", "R_top1", "R_top2", "R_top3", "mm_dist"):
                tb.add_scalar(f"eval/{k}", m[k], cstep)
            tb.flush()
            print(f"[evalmon] step {cstep}: FID {m['fid']:.3f}  R@3 {m['R_top3']:.3f}  "
                  f"Div {m['div']:.3f}  MMDist {m['mm_dist']:.3f}", flush=True)
            last_eval = cstep
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[evalmon] eval failed at step {step}: {e}", flush=True)
        if last_eval >= a.max_step:
            break
        time.sleep(a.poll)
    tb.close()
    print("[evalmon] done", flush=True)


if __name__ == "__main__":
    main()
