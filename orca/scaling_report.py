"""Stage A scaling report: does a bigger model close the joint-vs-specialist gap?

    python orca/scaling_report.py                      # reads ~/lcm-runs/stageA_n20k
    python orca/scaling_report.py <stageA results folder>

Reads every <size>_seed<N>/runs folder and writes SCALING.md and scaling.png
into the results folder (and prints the report).

The gap = joint model's CRPS / specialist's CRPS, combined over the 8 systems
with a geometric mean (so no single system dominates). 1.0 = the joint model
matches the specialists; higher = worse. CRPS (continuous ranked probability
score) measures how far the predicted range of futures is from what happened.

Success criteria, fixed before running (PROJECT_LOG.md, 2026-11-20), judged
against the 1.08M model trained under the same Stage A settings:
  gap shrinks   the gap at 8.5M is at least 0.10 below 1.08M
  gap closes    the gap is 1.05 or less at some size (a tie within noise)
  stop          the gap is still above 1.2 at 28M (capacity isn't the bottleneck)
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_curves import assess, curve  # noqa: E402

SYSTEMS = ["predator_prey", "boids", "kuramoto", "lorenz96", "nbody", "gene_reg", "neurons", "epidemic"]
HELD_OUT = ["neurons", "epidemic", "kuramoto"]
PHASE0_GAP = 1.34          # Phase 0, 1.08M, fp32, 5k steps, 2,000 episodes, 5 seeds (reference only)
SHRINK, CLOSE, STOP = 0.10, 1.05, 1.20


def size_value(name):
    return float(name.rstrip("M"))


def load_root(root):
    """{size: {seed: {tag: run}}} for every <size>_seed<N> folder that has results."""
    out = {}
    for d in sorted(os.listdir(root)):
        m = re.fullmatch(r"(.+M)_seed(\d+)", d)
        runs_dir = os.path.join(root, d, "runs")
        if not m or not os.path.isdir(runs_dir):
            continue
        runs = {f[:-5]: json.load(open(os.path.join(runs_dir, f))) for f in os.listdir(runs_dir)
                if f.endswith(".json")}
        out.setdefault(m.group(1), {})[int(m.group(2))] = runs
    return dict(sorted(out.items(), key=lambda kv: size_value(kv[0])))


def crps(runs, tag, system):
    return runs[tag]["eval"][system]["crps"]


def gap(runs):
    """Geometric mean over systems of joint / specialist CRPS, or None if a run is missing."""
    try:
        r = [crps(runs, "joint", s) / crps(runs, f"specialist_{s}", s) for s in SYSTEMS]
    except KeyError:
        return None
    return float(np.exp(np.mean(np.log(r))))


def summarize(data):
    """Per size: gaps per seed, per-system gaps, transfer results and overfitting counts."""
    out = {}
    for size, seeds in data.items():
        gaps = {k: gap(v) for k, v in seeds.items()}
        done = {k: g for k, g in gaps.items() if g is not None}
        per_system = {s: float(np.mean([crps(v, "joint", s) / crps(v, f"specialist_{s}", s)
                                        for k, v in seeds.items() if k in done])) for s in SYSTEMS} if done else {}
        zero_shot, few_shot = {}, {}
        for s in HELD_OUT:
            z = [(v[f"loo_{s}"]["eval"][s]["crps"], v[f"loo_{s}"]["eval"][s]["persist_crps"])
                 for v in seeds.values() if f"loo_{s}" in v]
            zero_shot[s] = (sum(a < b for a, b in z), len(z))
            f = [crps(v, f"scratch_{s}", s) / crps(v, f"finetune_{s}", s)
                 for v in seeds.values() if f"scratch_{s}" in v and f"finetune_{s}" in v]
            few_shot[s] = float(np.mean(f)) if f else None
        overfit = total = 0
        for v in seeds.values():
            for run in v.values():
                pts = curve(run)
                if pts:
                    total += 1
                    overfit += assess(pts)[3].startswith("OVERFITTING")
        n_runs = {k: len(v) for k, v in seeds.items()}
        out[size] = dict(gaps=done, n_runs=n_runs, per_system=per_system, zero_shot=zero_shot,
                         few_shot=few_shot, overfit=(overfit, total))
    return out


def mean_sd(values):
    v = np.array(list(values), float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def verdicts(summary):
    """The three pre-registered checks. Each is (label, outcome, explanation)."""
    means = {s: mean_sd(x["gaps"].values())[0] for s, x in summary.items() if x["gaps"]}
    out = []
    base, mid, big = means.get("1.08M"), means.get("8.5M"), means.get("28M")
    if base is not None and mid is not None:
        drop = base - mid
        sd_b, sd_m = (mean_sd(summary[s]["gaps"].values())[1] for s in ("1.08M", "8.5M"))
        n_b, n_m = len(summary["1.08M"]["gaps"]), len(summary["8.5M"]["gaps"])
        se = np.sqrt(sd_b ** 2 / n_b + sd_m ** 2 / n_m) if n_b > 1 and n_m > 1 else float("nan")
        noise = f"; 2x the seed noise = {2 * se:.3f}" if np.isfinite(se) else ""
        out.append(("Gap shrinks (8.5M at least 0.10 below 1.08M)", "YES" if drop >= SHRINK else "NO",
                    f"{base:.3f} -> {mid:.3f} (drop {drop:+.3f}{noise})"))
    else:
        out.append(("Gap shrinks", "NOT YET", "needs 1.08M and 8.5M results"))
    closed = [s for s, g in means.items() if g <= CLOSE]
    out.append(("Gap closes (1.05 or less at some size)", "YES" if closed else "NO",
                f"at {', '.join(closed)}" if closed else f"smallest gap {min(means.values()):.3f}"
                if means else "no results"))
    if big is not None:
        out.append(("Stop and rethink (gap still above 1.2 at 28M)", "YES - STOP" if big > STOP else "NO",
                    f"28M gap {big:.3f}"))
    else:
        out.append(("Stop and rethink", "NOT YET", "needs 28M results"))
    return out


def write_chart(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, gray, ink2, grid = "#2a78d6", "#9a9893", "#52514e", "#e6e5e0"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sizes = [s for s in summary if summary[s]["gaps"]]
    xs = [size_value(s) for s in sizes]
    for x, s in zip(xs, sizes):
        ax.scatter([x] * len(summary[s]["gaps"]), list(summary[s]["gaps"].values()), s=22, color=blue,
                   alpha=0.45, edgecolor="none")
    ax.plot(xs, [mean_sd(summary[s]["gaps"].values())[0] for s in sizes], color=blue, lw=2, marker="o", ms=7)
    ax.axhline(1.0, color=ink2, lw=1)
    ax.axhline(CLOSE, color=gray, lw=1, ls="--")
    ax.axhline(STOP, color=gray, lw=1, ls=":")
    ax.text(0.01, CLOSE, "closed (1.05)", va="bottom", ha="left", color=ink2, fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.text(0.01, STOP, "stop line at 28M (1.2)", va="bottom", ha="left", color=ink2, fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xticks(xs, sizes)
    ax.minorticks_off()
    ax.set_xlabel("model size (parameters)", color=ink2)
    ax.set_ylabel("joint / specialist error (1 = equal)", color=ink2)
    ax.set_title("Does a bigger model close the gap? (dots = seeds, line = mean)", loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=grid)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(root):
    data = load_root(root)
    if not data:
        sys.exit(f"No <size>_seed<N> result folders in {root}")
    s = summarize(data)
    lines = ["# Stage A scaling report", "",
             f"Results folder: `{root}`. Gap = joint / specialist CRPS, geometric mean over 8 systems "
             f"(1.0 = equal, higher = joint worse). Phase 0 reference gap: {PHASE0_GAP} "
             "(different settings, not used for the verdicts).", ""]

    lines += ["## Verdicts (criteria fixed before running)", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {a} | **{b}** | {c} |" for a, b, c in verdicts(s)]

    lines += ["", "## Gap by model size", "",
              "| Size | Seeds complete | Gap mean | sd | Per seed | Runs overfitting* |", "|---|---|---|---|---|---|"]
    for size, x in s.items():
        m, sd = mean_sd(x["gaps"].values()) if x["gaps"] else (float("nan"), float("nan"))
        per = ", ".join(f"s{k}: {g:.3f}" for k, g in sorted(x["gaps"].items()))
        lines.append(f"| {size} | {len(x['gaps'])} of {len(x['n_runs'])} | {m:.3f} | {sd:.3f} | {per} | "
                     f"{x['overfit'][0]} of {x['overfit'][1]} |")
    lines.append("\n*Validation NLL rose more than 0.05 after its best point; the best checkpoint was used anyway.")

    lines += ["", "## Gap per system (mean over seeds)", "", "| System | " + " | ".join(s) + " |",
              "|---|" + "---|" * len(s)]
    for sysname in SYSTEMS:
        lines.append(f"| {sysname} | " + " | ".join(f"{x['per_system'].get(sysname, float('nan')):.2f}"
                                                   for x in s.values()) + " |")

    lines += ["", "## Transfer (secondary)", "",
              "| Size | Zero-shot beats baseline (seeds) | Pretraining advantage, from-scratch / fine-tuned error |",
              "|---|---|---|"]
    for size, x in s.items():
        zs = ", ".join(f"{n} {k}/{t}" for n, (k, t) in x["zero_shot"].items())
        fs = ", ".join(f"{n} {v:.2f}x" for n, v in x["few_shot"].items() if v is not None)
        lines.append(f"| {size} | {zs} | {fs} |")

    chart = os.path.join(root, "scaling.png")
    write_chart(s, chart)
    lines += ["", "![gap vs size](scaling.png)", ""]
    text = "\n".join(lines) + "\n"
    open(os.path.join(root, "SCALING.md"), "w").write(text)
    print(text)
    print(f"Written: {root}/SCALING.md and {chart}")
    return s


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/lcm-runs/stageA_n20k"))
