"""Phase 0 experiment: does one model trained on many complex systems beat
specialists, and can it handle a system it has never seen?

    python experiment.py            # full run (~1-1.5 h on an M3 laptop)
    python experiment.py --quick    # small smoke run (~5 min)
    python experiment.py --seed 1 --out other_dir   # same data, different training randomness

Every finished run is cached in results/runs/, so an interrupted experiment
picks up where it stopped. The report is written to results/REPORT.md.
"""

import argparse
import json
import os
import time

import torch

from lcm.data import build_datasets, load
from lcm.evaluate import forecast_metrics, outcome_likelihood
from lcm.model import LCM
from lcm.simulators import DOMAINS
from lcm.train import train, get_device

HELD_OUT = ["neurons", "epidemic", "kuramoto"]   # systems hidden from the transfer models

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--out", default="results")
p.add_argument("--seed", type=int, default=0,
               help="training randomness (starting weights, batch order). The simulated data "
                    "and the true-odds replays stay fixed, so rerunning with another seed "
                    "measures how much results move by chance.")
args = p.parse_args()

if args.quick:
    CFG = dict(n_train=300, n_test=64, steps=300, few_shot_eps=32, few_shot_steps=100, eval_limit=32, n_ctx=16)
    args.out = args.out + "_quick"
else:
    CFG = dict(n_train=2000, n_test=256, steps=5000, few_shot_eps=64, few_shot_steps=500, eval_limit=128, n_ctx=64)

dev = get_device()
RUNS, CKPT = os.path.join(args.out, "runs"), os.path.join(args.out, "checkpoints")
DATA = os.path.join(args.out, "data")
for d in (RUNS, CKPT):
    os.makedirs(d, exist_ok=True)

print(f"device {dev} | seed {args.seed} | config {CFG}")
t0 = time.time()
build_datasets(DATA, CFG["n_train"], CFG["n_test"])
print(f"data ready ({time.time() - t0:.0f}s)")
train_sets = {n: load(DATA, n, "train", dev) for n in DOMAINS}
test_sets = {n: load(DATA, n, "test", dev) for n in DOMAINS}


def evaluate(model, domains):
    out = {}
    for n in domains:
        m = forecast_metrics(model, test_sets[n], limit=CFG["eval_limit"])
        m.update(outcome_likelihood(model, n, dev, n_ctx=CFG["n_ctx"]))
        out[n] = m
        print(f"    {n:14s} crps {m['crps']:.3f} (persist {m['persist_crps']:.3f})  "
              f"cover90 {m['cover90']:.2f}  prob_err {m['prob_error']:.3f} (base {m['prob_error_baserate']:.3f})")
    return out


def run(tag, domains_train, domains_eval, steps, data=None, init=None, seed=args.seed):
    """Train a model (or load it if done) and evaluate. Cached by tag."""
    path = os.path.join(RUNS, f"{tag}.json")
    ck = os.path.join(CKPT, f"{tag}.pt")
    if os.path.exists(path):
        print(f"[{tag}] cached")
        return json.load(open(path))
    print(f"[{tag}] training on {domains_train} for {steps} steps")
    torch.manual_seed(seed)
    model = LCM().to(dev)
    if init:
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{init}.pt"), map_location=dev))
    hist = []
    if steps:
        data = data or {n: train_sets[n] for n in domains_train}
        hist = train(model, data, steps, batch_size=32, seed=seed, log_every=max(steps // 5, 1), tag=tag,
                     lr=1e-4 if init else 3e-4)
    torch.save(model.state_dict(), ck)
    res = dict(tag=tag, trained_on=domains_train, steps=steps, seed=seed, loss=hist,
               eval=evaluate(model, domains_eval))
    json.dump(res, open(path, "w"))
    return res


# 1. one model, every system
run("joint", DOMAINS, DOMAINS, CFG["steps"])

# 2. one specialist per system, same training budget
for n in DOMAINS:
    run(f"specialist_{n}", [n], [n], CFG["steps"])

# 3. transfer: hide a system, then test zero-shot and with a little data
for n in HELD_OUT:
    others = [d for d in DOMAINS if d != n]
    run(f"loo_{n}", others, [n], CFG["steps"])
    few = {n: {k: v[: CFG["few_shot_eps"]] for k, v in train_sets[n].items()}}
    run(f"finetune_{n}", [n], [n], CFG["few_shot_steps"], data=few, init=f"loo_{n}")
    run(f"scratch_{n}", [n], [n], CFG["few_shot_steps"], data=few)

print(f"all runs done ({(time.time() - t0) / 60:.1f} min)")

from report import write_report  # noqa: E402
write_report(args.out, DOMAINS, HELD_OUT, CFG, test_sets, dev)
