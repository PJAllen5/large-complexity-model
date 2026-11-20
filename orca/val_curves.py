"""Read validation curves from finished runs and flag overfitting.

    python orca/val_curves.py ~/lcm-runs/stageA/8.5M_seed0 ~/lcm-runs/stageA/28M_seed0

For each run trained with validation (--n-val), shows the validation NLL
(lower is better) at evenly spaced points, where it was lowest, and how much
it rose after that. A model that keeps improving on training data while its
validation NLL climbs is memorising its training episodes (overfitting).
"""

import json
import os
import sys

RISE = 0.05   # NLL rise above the best point that counts as overfitting (judgement call)


def curve(run):
    """[(step, train_nll, val_nll)] from a run's loss history, or [] if it had no validation."""
    return [tuple(h) for h in run.get("loss", []) if len(h) == 3]


def assess(points, rise=RISE):
    """Best validation point, the rise after it, and a plain-language verdict."""
    best_step, _, best = min(points, key=lambda p: p[2])
    final_step, _, final = points[-1]
    up = final - best
    if up > rise:
        verdict = f"OVERFITTING: best at step {best_step:,}, then rose {up:+.3f}"
    elif best_step == final_step:
        verdict = "still improving at the end (could train longer)"
    else:
        verdict = f"flat after step {best_step:,} (rise {up:+.3f}, within {rise})"
    return best_step, best, up, verdict


def _step(s):
    return f"{s / 1000:g}k" if s >= 1000 else str(s)


def main(folders):
    for folder in folders:
        runs_dir = os.path.join(folder, "runs")
        if not os.path.isdir(runs_dir):
            print(f"{folder}: no runs yet")
            continue
        print(f"\n== {folder}")
        for f in sorted(os.listdir(runs_dir)):
            run = json.load(open(os.path.join(runs_dir, f)))
            pts = curve(run)
            if not pts:
                print(f"  {f[:-5]:26s} (no validation recorded)")
                continue
            shown = pts[:: max(len(pts) // 5, 1)]
            if shown[-1] != pts[-1]:
                shown.append(pts[-1])
            series = "  ".join(f"{_step(s)}:{v:.3f}" for s, _, v in shown)
            print(f"  {f[:-5]:26s} {series}")
            print(f"  {'':26s} -> {assess(pts)[3]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
