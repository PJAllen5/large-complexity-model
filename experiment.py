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
import sys
import os
import time

import torch

from lcm.data import build_datasets, load
from lcm.evaluate import forecast_metrics, outcome_likelihood
from lcm.model import LCM
from lcm.simulators import DOMAINS
from lcm.train import PRECISIONS, train, get_device

HELD_OUT = ["neurons", "epidemic", "kuramoto"]   # systems hidden from the transfer models

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true")
p.add_argument("--out", default="results")
p.add_argument("--seed", type=int, default=0,
               help="training randomness (starting weights, batch order). The simulated data "
                    "and the true-odds replays stay fixed, so rerunning with another seed "
                    "measures how much results move by chance.")
# Model size and training settings. Defaults reproduce Phase 0 exactly.
p.add_argument("--d", type=int, default=128, help="model width (numbers per token)")
p.add_argument("--layers", type=int, default=4, help="transformer blocks")
p.add_argument("--heads", type=int, default=4, help="attention heads")
p.add_argument("--precision", choices=PRECISIONS, default="fp32",
               help="fp32 = exact (Phase 0); tf32 / bf16 = faster on NVIDIA GPUs")
# Running on a cluster: one task per job, a shared data folder, a final report job.
p.add_argument("--data-dir", default=None, help="simulated data folder (default: <out>/data); "
               "share one between runs that use the same data")
p.add_argument("--prepare-data-only", action="store_true", help="only simulate the data, then stop")
p.add_argument("--only", default=None,
               help="comma-separated tasks to run: joint, specialist_<system>, transfer_<system> "
                    "(transfer = never-seen-it, then fine-tune and from-scratch on that held-out system)")
p.add_argument("--report-only", action="store_true", help="write the report from finished runs, train nothing")
for name, kind in [("steps", int), ("batch", int), ("lr", float), ("n_train", int), ("n_val", int),
                   ("few_shot_eps", int), ("few_shot_steps", int)]:
    p.add_argument(f"--{name.replace('_', '-')}", type=kind, default=None, dest=name)
args = p.parse_args()

if args.quick:
    CFG = dict(n_train=300, n_test=64, steps=300, few_shot_eps=32, few_shot_steps=100, eval_limit=32, n_ctx=16)
    args.out = args.out + "_quick"
else:
    CFG = dict(n_train=2000, n_test=256, steps=5000, few_shot_eps=64, few_shot_steps=500, eval_limit=128, n_ctx=64)
CFG.update(batch=32, lr=3e-4, n_val=0)
CFG.update({k: getattr(args, k) for k in ["steps", "batch", "lr", "n_train", "n_val", "few_shot_eps", "few_shot_steps"]
            if getattr(args, k) is not None})
MODEL = dict(d=args.d, layers=args.layers, heads=args.heads)

dev = get_device()
RUNS, CKPT = os.path.join(args.out, "runs"), os.path.join(args.out, "checkpoints")
DATA = args.data_dir or os.path.join(args.out, "data")
for d in (RUNS, CKPT):
    os.makedirs(d, exist_ok=True)

# Finished runs are cached by name, so never mix runs made with different settings
# in one folder: record the settings and refuse to continue if they differ.
SETTINGS = dict(cfg=CFG, model=MODEL, seed=args.seed, precision=args.precision)
cfg_path = os.path.join(args.out, "config.json")
if os.path.exists(cfg_path):
    old = json.load(open(cfg_path))
    old_settings = {k: old.get(k) for k in SETTINGS}
    if old_settings != json.loads(json.dumps(SETTINGS)):
        sys.exit(f"{args.out} holds runs made with different settings:\n  saved: {old_settings}\n"
                 f"  now:   {SETTINGS}\nUse a new --out folder.")


def save_atomic(save_fn, path):
    """Write to a temporary name, then rename, so parallel jobs never see half-written files."""
    tmp = f"{path}.{os.getpid()}.tmp"
    save_fn(tmp)
    os.replace(tmp, path)


save_atomic(lambda f: json.dump(dict(SETTINGS, device=str(dev), torch=torch.__version__), open(f, "w"), indent=1),
            cfg_path)

print(f"device {dev} | seed {args.seed} | precision {args.precision} | model {MODEL} | config {CFG}")
t0 = time.time()
build_datasets(DATA, CFG["n_train"], CFG["n_test"], n_val=CFG["n_val"])
print(f"data ready ({time.time() - t0:.0f}s)")
if args.prepare_data_only:
    sys.exit(0)
train_sets = {n: load(DATA, n, "train", dev) for n in DOMAINS}
test_sets = {n: load(DATA, n, "test", dev) for n in DOMAINS}
val_sets = {n: load(DATA, n, "val", dev) for n in DOMAINS} if CFG["n_val"] else {}


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
    model = LCM(**MODEL).to(dev)
    if init:
        model.load_state_dict(torch.load(os.path.join(CKPT, f"{init}.pt"), map_location=dev))
    hist = []
    if steps:
        data = data or {n: train_sets[n] for n in domains_train}
        val = {n: val_sets[n] for n in data} if val_sets else None
        hist = train(model, data, steps, batch_size=CFG["batch"], seed=seed, log_every=max(steps // 20, 1),
                     tag=tag, lr=1e-4 if init else CFG["lr"], val=val, precision=args.precision)
    save_atomic(lambda f: torch.save(model.state_dict(), f), ck)
    # with validation on, train() keeps the weights from the best-validation step
    best_step = min((h for h in hist if len(h) == 3), key=lambda h: h[2], default=(None,))[0]
    res = dict(tag=tag, trained_on=domains_train, steps=steps, seed=seed, best_step=best_step, loss=hist,
               eval=evaluate(model, domains_eval))
    save_atomic(lambda f: json.dump(res, open(f, "w")), path)
    return res


def run_task(task):
    if task == "joint":                                   # one model, every system
        run("joint", DOMAINS, DOMAINS, CFG["steps"])
    elif task.startswith("specialist_"):                  # one system only, same budget
        n = task[len("specialist_"):]
        run(task, [n], [n], CFG["steps"])
    elif task.startswith("transfer_"):                    # hide a system, then zero-shot and few-shot
        n = task[len("transfer_"):]
        others = [d for d in DOMAINS if d != n]
        run(f"loo_{n}", others, [n], CFG["steps"])
        few = {n: {k: v[: CFG["few_shot_eps"]] for k, v in train_sets[n].items()}}
        run(f"finetune_{n}", [n], [n], CFG["few_shot_steps"], data=few, init=f"loo_{n}")
        run(f"scratch_{n}", [n], [n], CFG["few_shot_steps"], data=few)


ALL_TASKS = ["joint"] + [f"specialist_{n}" for n in DOMAINS] + [f"transfer_{n}" for n in HELD_OUT]
RUN_TAGS = (["joint"] + [f"specialist_{n}" for n in DOMAINS]
            + [f"{k}_{n}" for n in HELD_OUT for k in ("loo", "finetune", "scratch")])

if args.only:
    tasks = args.only.split(",")
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        sys.exit(f"unknown task(s) {unknown}; choose from {ALL_TASKS}")
elif args.report_only:
    tasks = []
    missing = [t for t in RUN_TAGS if not os.path.exists(os.path.join(RUNS, f"{t}.json"))]
    if missing:
        sys.exit(f"can't write the report: runs not finished yet: {missing}")
else:
    tasks = ALL_TASKS
for task in tasks:
    run_task(task)

if args.only:
    print(f"tasks {tasks} done ({(time.time() - t0) / 60:.1f} min); report skipped (use --report-only "
          "once every task has finished)")
    sys.exit(0)

print(f"all runs done ({(time.time() - t0) / 60:.1f} min)")

from report import write_report  # noqa: E402
write_report(args.out, DOMAINS, HELD_OUT, CFG, test_sets, dev, model_kwargs=MODEL)
