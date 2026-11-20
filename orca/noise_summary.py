"""Summarise run-to-run noise across several seeds of the same experiment.

    python orca/noise_summary.py results_orca ~/lcm-runs/phase0_seed1 ~/lcm-runs/phase0_seed2 ...

Each folder is one full Phase 0 run that differs only in training seed. Reports:
  - per run: mean CRPS, standard deviation (sd), relative spread (cv = sd / mean),
    and min-max across seeds
  - per run type: typical (median) relative spread
  - each headline finding: in how many seeds it holds
  - pretraining's few-shot advantage (from-scratch error / fine-tuned error) per seed
  - a rough number of seeds needed to detect a 10% or 20% improvement
    (from the worst spread of each run type, so it covers every run)

CRPS = continuous ranked probability score: how far the predicted range of
futures is from what happened (lower is better).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare import HELD_OUT, headline_items, load  # noqa: E402

GROUPS = [("specialist", "specialist_"), ("joint", "joint"), ("zero-shot", "loo_"),
          ("fine-tune (64 eps)", "finetune_"), ("from scratch (64 eps)", "scratch_")]


def group_of(tag):
    for name, prefix in GROUPS:
        if tag.startswith(prefix):
            return name
    return "other"


def seeds_needed(cv, effect):
    """Seeds per model so a relative improvement `effect` is ~2 standard errors.

    Comparing two averages of n seeds each, the standard error of the difference
    is about cv * sqrt(2 / n). Requiring effect >= 2 * that gives n >= 8 cv^2 / effect^2.
    A rough rule of thumb, not a formal power analysis.
    """
    return max(2, int(np.ceil(8 * cv ** 2 / effect ** 2)))


def summarize(folders):
    runs = [load(f) for f in folders]
    seeds = [r["joint"].get("seed", 0) if "joint" in r else None for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds among folders: {seeds}")
    tags = sorted(set.intersection(*[set(r) for r in runs]))

    per_run = []   # (group, tag, dom, values)
    for tag in tags:
        for dom in runs[0][tag]["eval"]:
            vals = np.array([r[tag]["eval"][dom]["crps"] for r in runs])
            per_run.append((group_of(tag), tag, dom, vals))

    group_cv = {}
    for name, _ in GROUPS:
        cvs = [v.std(ddof=1) / v.mean() for g, _, _, v in per_run if g == name]
        if cvs:
            group_cv[name] = (float(np.median(cvs)), float(np.max(cvs)))

    findings = {}
    for r in runs:
        for q, item, a, b in headline_items(r):
            findings.setdefault((q, item), []).append(bool(a < b))

    advantage = {n: [r[f"scratch_{n}"]["eval"][n]["crps"] / r[f"finetune_{n}"]["eval"][n]["crps"] for r in runs]
                 for n in HELD_OUT}
    return dict(seeds=seeds, per_run=per_run, group_cv=group_cv, findings=findings, advantage=advantage)


def main(folders):
    s = summarize(folders)
    n = len(folders)
    print(f"{n} runs, seeds {s['seeds']}\n")

    print(f"{'run':26s} {'system':14s} {'mean':>7s} {'sd':>7s} {'cv':>6s} {'min':>7s} {'max':>7s}")
    for g, tag, dom, v in s["per_run"]:
        print(f"{tag:26s} {dom:14s} {v.mean():7.3f} {v.std(ddof=1):7.3f} {v.std(ddof=1) / v.mean():6.1%} "
              f"{v.min():7.3f} {v.max():7.3f}")

    print(f"\n{'run type':24s} {'typical cv':>11s} {'worst cv':>9s} {'seeds for 20%':>14s} {'seeds for 10%':>14s}")
    for name, (med, worst) in s["group_cv"].items():
        print(f"{name:24s} {med:11.1%} {worst:9.1%} {seeds_needed(worst, 0.20):14d} {seeds_needed(worst, 0.10):14d}")
    print("(cv = sd / mean across seeds. 'Seeds for X%' = seeds per model to tell an X% average "
          "improvement from chance for every run of that type, using the worst cv; rough rule of thumb.)")

    print(f"\n{'finding':30s} {'item':14s} {'holds in':>9s}")
    for (q, item), held in s["findings"].items():
        k = sum(held)
        mark = "" if k in (0, n) else "  <-- seed-dependent"
        print(f"{q:30s} {item:14s} {f'{k} of {n}':>9s}{mark}")

    print("\nPretraining advantage on few-shot (from-scratch error / fine-tuned error, >1 = pretraining helps):")
    for name, vals in s["advantage"].items():
        print(f"  {name:10s} " + "  ".join(f"{x:.2f}x" for x in vals) + f"   mean {np.mean(vals):.2f}x")
    quality = "a rough estimate" if n < 4 else "a usable estimate"
    print(f"\nWith {n} seeds this is {quality}; more seeds narrow it further.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
