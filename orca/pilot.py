"""Scale-up pilot: how fast and how big is each model size on this GPU?

    python orca/pilot.py                 # on an ORCA GPU node (via orca/pilot.sbatch)
    python orca/pilot.py --tiny          # quick local check of the script itself

Part 1 (speed): for each model size x precision x batch size, time training
steps on synthetic data and record peak GPU memory. Projects the GPU-hours for
the scale-up: 2 new sizes x 3 seeds x 18 runs.

Part 2 (accuracy): briefly train the 8.5M model on real simulated data in each
precision, from the same starting weights and batches, and compare validation
NLL (lower is better). If tf32 or bf16 is much worse than fp32, faster maths
isn't safe to use.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lcm.data import generate  # noqa: E402
from lcm.model import LCM  # noqa: E402
from lcm.simulators import DOMAINS  # noqa: E402
from lcm.train import PRECISIONS, autocast_for, get_device, set_precision, train, validation_nll  # noqa: E402

SIZES = {"1.08M": dict(d=128, layers=4, heads=4), "8.5M": dict(d=256, layers=8, heads=8),
         "28M": dict(d=384, layers=12, heads=8)}
RUNS_PER_SIZE = 3 * 18          # seeds x runs per Phase 0 experiment

p = argparse.ArgumentParser()
p.add_argument("--tiny", action="store_true", help="tiny settings for a local smoke test")
p.add_argument("--plan-steps", type=int, default=20000, help="assumed training steps per scale-up run")
args = p.parse_args()

dev = get_device()
sizes = {"1.08M": SIZES["1.08M"]} if args.tiny else SIZES
precisions = PRECISIONS if dev.type == "cuda" else ("fp32",)
batches = (8,) if args.tiny else (32, 128)
n_time = 3 if args.tiny else 20
if dev.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
else:
    print(f"device {dev} (not CUDA: only fp32 is meaningful here)")


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- part 1: speed and memory
print(f"\n{'size':6s} {'params':>8s} {'precision':>9s} {'batch':>5s} {'s/step':>8s} {'peak GB':>8s} "
      f"{'GPU-h for scale-up*':>20s}")
results = {}
for name, kw in sizes.items():
    for prec in precisions:
        for B in batches:
            set_precision(prec)
            torch.manual_seed(0)
            m = LCM(**kw).to(dev)
            opt = torch.optim.AdamW(m.parameters())
            x = torch.randn(B, 64, 12, 4, device=dev)
            em, fm = torch.ones(B, 12, device=dev), torch.ones(B, 4, device=dev)
            adj, st = torch.ones(B, 12, 12, device=dev), torch.zeros(B, 8, device=dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                for i in range(n_time + 3):
                    if i == 3:
                        sync()
                        t = time.time()
                    with autocast_for(prec, dev):
                        loss = m.loss(x, em, fm, adj, st)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                sync()
                step = (time.time() - t) / n_time
                peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
                # steps per run scale inversely with batch (same number of episodes seen)
                hours = step * args.plan_steps * (32 / B) * RUNS_PER_SIZE / 3600
                results[(name, prec, B)] = step
                params = sum(q.numel() for q in m.parameters()) / 1e6
                print(f"{name:6s} {params:7.2f}M {prec:>9s} {B:5d} {step:8.4f} {peak:8.2f} {hours:20.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{name:6s} {'':>8s} {prec:>9s} {B:5d}   out of GPU memory")
            del m, opt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
print(f"*{args.plan_steps:,} steps at batch 32 (fewer at bigger batches), x {RUNS_PER_SIZE} runs per size. "
      "Few-shot runs are shorter, so this overestimates.")

# ---------------------------------------------------------------- part 2: does faster maths hurt accuracy?
acc_size = "1.08M" if args.tiny else "8.5M"
n_eps, n_steps = (16, 20) if args.tiny else (300, 1500)
print(f"\nAccuracy check: {acc_size} model, {n_steps} steps on {n_eps} episodes per system, "
      "same start and batches for each precision")
train_sets, val_sets = {}, {}
for i, n in enumerate(DOMAINS):
    train_sets[n] = {k: torch.from_numpy(v).to(dev) for k, v in generate(n, n_eps, 1000 + i).items()}
    val_sets[n] = {k: torch.from_numpy(v).to(dev) for k, v in generate(n, n_eps // 4 or 4, 2000 + i).items()}
val_scores = {}
for prec in precisions:
    torch.manual_seed(0)
    m = LCM(**SIZES[acc_size]).to(dev)
    t = time.time()
    train(m, train_sets, n_steps, batch_size=32, seed=0, log_every=n_steps, tag=prec, precision=prec)
    val_scores[prec] = validation_nll(m, val_sets, precision="fp32")   # judged in exact maths
    print(f"  {prec}: validation NLL {val_scores[prec]:.3f}  ({time.time() - t:.0f}s)")
if "fp32" in val_scores:
    for prec, v in val_scores.items():
        if prec != "fp32":
            print(f"  {prec} vs fp32: {v - val_scores['fp32']:+.3f} NLL "
                  "(a gap of a few hundredths is within normal run-to-run variation)")
sys.exit(0)
