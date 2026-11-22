"""Pretrain a model on real epidemics (Project Tycho), alone or mixed with the simulators.

    python pretrain.py --variant tycho    --seed 0 --d 256 --layers 8 --heads 8 --out ~/lcm-runs/pretrain_tycho/8.5M_seed0
    python pretrain.py --variant simtycho --seed 0 --d 256 --layers 8 --heads 8 --out ~/lcm-runs/pretrain_simtycho/8.5M_seed0

  tycho     the Tycho diseases only, each sampled equally
  simtycho  the 8 simulators plus Tycho as a 9th system (its diseases share one
            system's share of every batch: 1/6 each)

Writes <out>/checkpoints/joint.pt (the best-validation weights; the same layout
as the Stage A joint runs, so orca/phase1.sbatch can fine-tune from it) and
<out>/pretrain.json. Validation every --every steps; stops after --patience
checks without a new best. For simtycho, Tycho's validation windows are pooled
into one "tycho" set, so validation weighs it as one system of nine.
"""

import argparse
import json
import os
import sys
import time

import torch

from lcm import tycho
from lcm.data import load
from lcm.model import LCM
from lcm.simulators import DOMAINS
from lcm.train import PRECISIONS, get_device, train

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--variant", required=True, choices=["tycho", "simtycho"])
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--d", type=int, default=256)
p.add_argument("--layers", type=int, default=8)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--steps", type=int, default=None, help="default 160000 (tycho) or 180000 (simtycho: 9 x 20k)")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--precision", choices=PRECISIONS, default="fp32")
p.add_argument("--every", type=int, default=4000, help="validation spacing in steps")
p.add_argument("--patience", type=int, default=12)
p.add_argument("--tycho", default=os.path.expanduser("~/lcm-runs/tycho"), help="built Tycho windows")
p.add_argument("--sim-data", default=os.path.expanduser("~/lcm-runs/stageA_n20k/data"),
               help="simulated data (Stage A: 20,000 train / 256 val per system)")
args = p.parse_args()

steps = args.steps or (160000 if args.variant == "tycho" else 180000)
ckpt = os.path.join(args.out, "checkpoints", "joint.pt")
if os.path.exists(ckpt):
    print(f"{ckpt} exists; skipping")
    sys.exit(0)
os.makedirs(os.path.dirname(ckpt), exist_ok=True)
dev = get_device()

train_sets = tycho.load(args.tycho, "train", dev)
val_t = tycho.load(args.tycho, "val", dev)
if not train_sets:
    sys.exit(f"no Tycho windows in {args.tycho}; run: python -m lcm.tycho fetch/build {args.tycho}")
weights = {k: 1.0 for k in train_sets}
if args.variant == "tycho":
    val_sets = val_t
else:
    share = 1.0 / len(train_sets)
    weights = {k: share for k in train_sets}
    val_sets = {"tycho": {k: torch.cat([v[k] for v in val_t.values()]) for k in next(iter(val_t.values()))}}
    for n in DOMAINS:
        train_sets[n] = load(args.sim_data, n, "train", dev)
        val_sets[n] = load(args.sim_data, n, "val", dev)
        weights[n] = 1.0
print(f"{args.variant}: {steps} steps on {dev}; training sets "
      f"{ {k: len(v['x']) for k, v in train_sets.items()} }; weights {weights}")

torch.manual_seed(args.seed)
model = LCM(d=args.d, layers=args.layers, heads=args.heads).to(dev)
val_log, t0 = [], time.time()
hist = train(model, train_sets, steps, batch_size=args.batch, lr=args.lr, seed=args.seed, tag=args.variant,
             val=val_sets, precision=args.precision, weights=weights, val_log=val_log,
             eval_at=range(args.every, steps + 1, args.every), patience=args.patience)
tmp = f"{ckpt}.{os.getpid()}.tmp"
torch.save(model.state_dict(), tmp)
os.replace(tmp, ckpt)
best_step = min(hist, key=lambda h: h[2])[0]
info = dict(variant=args.variant, seed=args.seed, model=dict(d=args.d, layers=args.layers, heads=args.heads),
            steps=steps, stopped_at=hist[-1][0], best_step=best_step, batch=args.batch, lr=args.lr,
            precision=args.precision, every=args.every, patience=args.patience, weights=weights,
            minutes=round((time.time() - t0) / 60, 1), loss=hist, val_by_system=val_log)
json.dump(info, open(os.path.join(args.out, "pretrain.json"), "w"), indent=1)
print(f"best step {best_step}, stopped at {hist[-1][0]}, {info['minutes']} min; wrote {ckpt}")
