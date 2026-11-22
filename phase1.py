"""Phase 1: does pretraining on simulated systems help forecast a real one?

One run = one model (or the baselines) on one real dataset and data regime:

    python phase1.py --dataset flu --regime small --mode baselines --out ~/lcm-runs/phase1
    python phase1.py --dataset flu --regime small --mode scratch  --seed 0 --d 256 --layers 8 --heads 8 ...
    python phase1.py --dataset flu --regime small --mode finetune --seed 0 --d 256 --layers 8 --heads 8 \\
        --init ~/lcm-runs/stageA_joint160k/8.5M_seed0/checkpoints/joint.pt ...

  scratch   a fresh model trained only on the real training windows
  finetune  the simulator-pretrained joint model, then trained on the same windows
  baselines persistence (last value) and seasonal-naive (one period earlier)

Regimes: `small` uses only the dataset's small-data training range, `full` all of
it. Both use the same validation windows (best checkpoint kept) and the same
test windows. Results go to <out>/<dataset>_<regime>/<mode>[_<size>_seed<N>].json;
finished runs are skipped.

Validation (for keeping the best checkpoint) runs at step 0, every 25 steps up to
step 1,000, then every 250. Fine-tuned models peaked before the first check in the
first flu run, which used every 250 only (PROJECT_LOG.md 2026-11-21). Fine-tuned
runs also record the pretrained model's test scores before any fine-tuning
("test_step0"), and every run records each test window's CRPS and 90% coverage.
"""

import argparse
import json
import os
import sys
import time

import torch

from lcm.data import T_CTX
from lcm.evaluate import forecast_metrics
from lcm.model import LCM
from lcm.real import REAL, point_crps, seasonal_episodes, split_starts, to_episodes
from lcm.train import PRECISIONS, get_device, train

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset", required=True, choices=sorted(REAL))
p.add_argument("--regime", required=True, choices=["small", "full"])
p.add_argument("--mode", required=True, choices=["scratch", "finetune", "baselines"])
p.add_argument("--out", default=os.path.expanduser("~/lcm-runs/phase1_v2"))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--size", default=None, help="label for the model size in file names (e.g. 8.5M)")
p.add_argument("--init", default=None, help="finetune: the pretrained joint model's checkpoint (.pt)")
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=None, help="default 3e-4 from scratch, 1e-4 fine-tuning")
p.add_argument("--precision", choices=PRECISIONS, default="fp32")
p.add_argument("--dense-until", type=int, default=1000, help="validate every --dense-every steps up to here")
p.add_argument("--dense-every", type=int, default=25)
p.add_argument("--every", type=int, default=250, help="validation spacing after --dense-until")
args = p.parse_args()

if args.mode == "finetune" and not args.init:
    sys.exit("--mode finetune needs --init <pretrained checkpoint>")
if args.mode != "finetune" and args.init:
    sys.exit("--init only applies to --mode finetune")

ds = REAL[args.dataset]()
folder = os.path.join(args.out, f"{args.dataset}_{args.regime}")
os.makedirs(folder, exist_ok=True)
size = args.size or f"d{args.d}x{args.layers}"
name = "baselines" if args.mode == "baselines" else f"{args.mode}_{size}_seed{args.seed}"
path = os.path.join(folder, f"{name}.json")
if os.path.exists(path):
    print(f"{path} exists; skipping")
    sys.exit(0)

dev = get_device()
lo, hi = ds["splits"]["small_train" if args.regime == "small" else "train"]
episodes = {s: to_episodes(ds["values"], ds["adj"], split_starts(*r))
            for s, r in [("train", (lo, hi)), ("val", ds["splits"]["val"]), ("test", ds["splits"]["test"])]}
tensors = {s: {k: torch.from_numpy(v).to(dev) for k, v in e.items()} for s, (e, _) in episodes.items()}
counts = {s: len(starts) for s, (_, starts) in episodes.items()}
print(f"{args.dataset} / {args.regime}: windows {counts} on {dev}")


def save(res):
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    print(f"wrote {path}")


res = dict(dataset=args.dataset, regime=args.regime, mode=args.mode, windows=counts,
           train_range=[int(ds["times"][lo]), int(ds["times"][hi - 1])])
if args.mode == "baselines":
    test = tensors["test"]
    res["persist_crps"] = point_crps(test)
    if ds.get("period"):
        alt, kept = seasonal_episodes(ds["values"], ds["adj"], episodes["test"][1], ds["period"])
        if kept == episodes["test"][1]:
            res["seasonal_crps"] = point_crps(test, torch.from_numpy(alt["x"]).to(dev))
        else:
            print(f"seasonal baseline needs lagged data for every test window; skipped ({len(kept)} usable)")
    save(res)
    sys.exit(0)

torch.manual_seed(args.seed)
model = LCM(d=args.d, layers=args.layers, heads=args.heads).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev))
lr = args.lr or (1e-4 if args.init else 3e-4)
val_steps = sorted({0} | set(range(args.dense_every, min(args.dense_until, args.steps) + 1, args.dense_every))
                   | set(range(args.dense_until + args.every, args.steps + 1, args.every)) | {args.steps})
t0 = time.time()
step0 = None
if args.init:                                         # the pretrained model before any fine-tuning
    step0 = forecast_metrics(model, tensors["test"], limit=counts["test"])
    print(f"  step 0 (no fine-tuning): test crps {step0['crps']:.3f}  cover90 {step0['cover90']:.2f}")
hist = train(model, {args.dataset: tensors["train"]}, args.steps, batch_size=args.batch, lr=lr, seed=args.seed,
             tag=name, val={args.dataset: tensors["val"]}, precision=args.precision, eval_at=val_steps)
best_step = min(hist, key=lambda h: h[2])[0]
metrics = forecast_metrics(model, tensors["test"], limit=counts["test"], per_window=True)
print(f"  test crps {metrics['crps']:.3f} (persist {metrics['persist_crps']:.3f})  cover90 {metrics['cover90']:.2f}")
res.update(seed=args.seed, size=size, model=dict(d=args.d, layers=args.layers, heads=args.heads), init=args.init,
           steps=args.steps, batch=args.batch, lr=lr, precision=args.precision, best_step=best_step, loss=hist,
           minutes=round((time.time() - t0) / 60, 1), val_steps=val_steps, test=metrics,
           test_forecast_start=[int(ds["times"][s + T_CTX]) for s in episodes["test"][1]])
if step0 is not None:
    res["test_step0"] = step0
save(res)
