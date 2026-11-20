"""Compare two Phase 0 result folders (e.g. laptop vs ORCA).

    python orca/compare.py results results_orca

The simulated data is identical (same seeds), so differences come only from
the GPU doing floating-point math differently and from random sampling.

Pass criterion (decided before running, see PROJECT_LOG.md):
  1. every forecast score (CRPS) within 10% of the reference run, and
  2. the same three headline answers as the reference report.
CRPS = continuous ranked probability score: how far the whole predicted
range of futures is from what happened (lower is better).
"""

import json
import os
import sys

TOL = 0.10
HELD_OUT = ["neurons", "epidemic", "kuramoto"]


def load(folder):
    d = os.path.join(folder, "runs")
    if not os.path.isdir(d):
        sys.exit(f"No runs found in {d}")
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")}


def headline(runs):
    """The three yes/no counts reported in REPORT.md."""
    J = runs["joint"]["eval"]
    doms = list(J)
    spec = {n: runs[f"specialist_{n}"]["eval"][n] for n in doms}
    joint_wins = sum(J[n]["crps"] < spec[n]["crps"] for n in doms)
    zero_shot = sum(runs[f"loo_{n}"]["eval"][n]["crps"] < runs[f"loo_{n}"]["eval"][n]["persist_crps"] for n in HELD_OUT)
    few_shot = sum(runs[f"finetune_{n}"]["eval"][n]["crps"] < runs[f"scratch_{n}"]["eval"][n]["crps"] for n in HELD_OUT)
    return {"joint beats specialist": f"{joint_wins} of {len(doms)}",
            "zero-shot beats baseline": f"{zero_shot} of {len(HELD_OUT)}",
            "pretraining helps few-shot": f"{few_shot} of {len(HELD_OUT)}"}


def main(ref_dir, new_dir):
    ref, new = load(ref_dir), load(new_dir)
    common = sorted(set(ref) & set(new))
    missing = sorted(set(ref) - set(new))
    print(f"{'run':28s} {'system':14s} {'CRPS ref':>9s} {'CRPS new':>9s} {'diff':>7s}")
    n_ok = n_all = 0
    for tag in common:
        for dom, m in ref[tag]["eval"].items():
            if dom not in new[tag]["eval"]:
                continue
            a, b = m["crps"], new[tag]["eval"][dom]["crps"]
            rel = b / a - 1
            ok = abs(rel) <= TOL
            n_ok += ok
            n_all += 1
            print(f"{tag:28s} {dom:14s} {a:9.3f} {b:9.3f} {rel:+7.1%}{'' if ok else '  <-- outside 10%'}")

    print(f"\nScores within {TOL:.0%}: {n_ok} of {n_all}")
    if missing:
        print(f"Runs missing from {new_dir}: {', '.join(missing)}")

    same = True
    if not missing:
        h_ref, h_new = headline(ref), headline(new)
        print(f"\n{'headline question':28s} {'ref':>8s} {'new':>8s}")
        for k in h_ref:
            same &= h_ref[k] == h_new[k]
            print(f"{k:28s} {h_ref[k]:>8s} {h_new[k]:>8s}{'' if h_ref[k] == h_new[k] else '  <-- differs'}")

    passed = n_all > 0 and n_ok == n_all and same and not missing
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'} "
          f"(criterion: all CRPS within {TOL:.0%} and the same three headline answers)")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
