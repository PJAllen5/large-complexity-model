"""Does the joint model still beat the specialists when both get 160k training steps?

    python orca/convergence_report.py                  # default folders below
    python orca/convergence_report.py --joint ~/lcm-runs/stageA_joint160k \\
        --specialists ~/lcm-runs/stageA_spec160k --specialists-20k ~/lcm-runs/stageA_n20k

Joint models come from --joint, the 160k-step specialists from --specialists.
The 20k-step specialists (--specialists-20k, optional) show how much the
specialists gained from the extra training. Writes CONVERGENCE.md into the
--specialists folder and prints it.

Gap = joint CRPS / specialist CRPS, geometric mean over the 8 systems (as in
scaling_report.py). Verdict per size, fixed before running (PROJECT_LOG.md,
2026-11-20); the seed noise of the gap is about 0.01:
  joint still wins   gap 0.97 or less
  tie                gap between 0.97 and 1.03 (sharing costs nothing)
  specialists win    gap above 1.03
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling_report import SYSTEMS, combine, crps, load_root, mean_sd, summarize  # noqa: E402
from val_curves import assess, curve  # noqa: E402

WIN, LOSE = 0.97, 1.03


def verdict(g):
    if g <= WIN:
        return "JOINT STILL WINS"
    if g <= LOSE:
        return "TIE"
    return "SPECIALISTS WIN"


def specialist_gain(long_data, short_data):
    """Per size and system: mean 160k / 20k specialist CRPS over the seeds both have."""
    out = {}
    for size, seeds in long_data.items():
        per = {}
        for s in SYSTEMS:
            tag = f"specialist_{s}"
            r = [crps(v, tag, s) / crps(short_data[size][k], tag, s) for k, v in seeds.items()
                 if tag in v and tag in short_data.get(size, {}).get(k, {})]
            if r:
                per[s] = float(np.mean(r))
        out[size] = per
    return out


def curve_states(seeds):
    """(overfitting, still improving at the end, total) over the specialist runs."""
    over = improving = total = 0
    for runs in seeds.values():
        for tag, run in runs.items():
            pts = curve(run)
            if not tag.startswith("specialist_") or not pts:
                continue
            total += 1
            v = assess(pts)[3]
            over += v.startswith("OVERFITTING")
            improving += v.startswith("still improving")
    return over, improving, total


def report(joint_root, spec_root, short_root=None):
    for folder in (joint_root, spec_root, short_root):
        if folder and not os.path.isdir(folder):
            sys.exit(f"No such results folder: {folder}")
    spec = load_root(spec_root)
    s = summarize(combine(load_root(joint_root), spec))
    if not s:
        sys.exit(f"No seeds with both a joint run in {joint_root} and specialists in {spec_root}")
    old = summarize(combine(load_root(joint_root), load_root(short_root))) if short_root else {}
    gain = specialist_gain(spec, load_root(short_root)) if short_root else {}

    lines = ["# Joint vs specialists at equal training (160k steps each)", "",
             f"Joint models from `{joint_root}`; 160k-step specialists from `{spec_root}`"
             + (f"; 20k-step specialists from `{short_root}`." if short_root else "."),
             "Gap = joint / specialist CRPS, geometric mean over 8 systems (below 1 = joint better).", "",
             "## Verdict (criteria fixed before running)", "",
             f"Joint still wins: gap {WIN} or less. Tie: {WIN}-{LOSE}. Specialists win: above {LOSE}.", "",
             "| Size | Seeds complete | Gap mean | sd | Per seed | Verdict |"
             + (" Gap vs 20k specialists |" if old else ""),
             "|---|---|---|---|---|---|" + ("---|" if old else "")]
    for size, x in s.items():
        if not x["gaps"]:
            lines.append(f"| {size} | 0 of {len(x['n_runs'])} | - | - | - | NOT YET |" + (" - |" if old else ""))
            continue
        m, sd = mean_sd(x["gaps"].values())
        per = ", ".join(f"s{k}: {g:.3f}" for k, g in sorted(x["gaps"].items()))
        row = f"| {size} | {len(x['gaps'])} of {len(x['n_runs'])} | {m:.3f} | {sd:.3f} | {per} | **{verdict(m)}** |"
        if old:
            og = old.get(size, {}).get("gaps")
            row += f" {mean_sd(og.values())[0]:.3f} |" if og else " - |"
        lines.append(row)

    lines += ["", "## Gap per system (mean over seeds)", "", "| System | " + " | ".join(s) + " |",
              "|---|" + "---|" * len(s)]
    for name in SYSTEMS:
        lines.append(f"| {name} | " + " | ".join(f"{x['per_system'].get(name, float('nan')):.2f}"
                                                for x in s.values()) + " |")

    if gain:
        lines += ["", "## How much the specialists gained from 8x training",
                  "", "160k-step specialist CRPS / 20k-step specialist CRPS (below 1 = improved).", "",
                  "| System | " + " | ".join(gain) + " |", "|---|" + "---|" * len(gain)]
        for name in SYSTEMS:
            lines.append(f"| {name} | " + " | ".join(f"{g[name]:.2f}" if name in g else "-"
                                                    for g in gain.values()) + " |")

    lines += ["", "## Specialist validation curves", "",
              "| Size | Overfitting* | Still improving at the end | Runs |", "|---|---|---|---|"]
    for size, seeds in spec.items():
        o, i, t = curve_states(seeds)
        lines.append(f"| {size} | {o} | {i} | {t} |")
    lines.append("\n*Validation NLL rose more than 0.05 after its best point; the best checkpoint was used anyway.")

    text = "\n".join(lines) + "\n"
    open(os.path.join(spec_root, "CONVERGENCE.md"), "w").write(text)
    print(text)
    print(f"Written: {spec_root}/CONVERGENCE.md")
    return s


if __name__ == "__main__":
    import argparse
    home = os.path.expanduser("~/lcm-runs")
    p = argparse.ArgumentParser(description="Joint vs specialists at equal (160k-step) training")
    p.add_argument("--joint", default=os.path.join(home, "stageA_joint160k"))
    p.add_argument("--specialists", default=os.path.join(home, "stageA_spec160k"))
    p.add_argument("--specialists-20k", default=os.path.join(home, "stageA_n20k"),
                   help="20k-step specialists, to show how much they gained ('' to skip)")
    a = p.parse_args()
    report(a.joint, a.specialists, a.specialists_20k or None)
