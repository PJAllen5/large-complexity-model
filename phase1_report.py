"""Phase 1 report: pretrained vs from-scratch vs simple baselines on a real dataset.

    python phase1_report.py ~/lcm-runs/phase1 flu

Reads <root>/<dataset>_small and <root>/<dataset>_full, writes
<root>/<dataset>_PHASE1.md and prints it. CRPS is in the model's normalised
units (lower is better); for the point baselines it equals the mean absolute error.

Verdict (small-data regime, per model size), fixed before running
(PROJECT_LOG.md, 2026-11-21); all four must hold:
  pretraining helps   fine-tuned CRPS at least 10% below from-scratch (mean over seeds)
  beats persistence   fine-tuned CRPS below "same as last step"
  beats seasonal      fine-tuned CRPS below "same time last period" (if the dataset has a period)
  calibrated          fine-tuned 90% range holds the truth 80-95% of the time
The full-data regime is reported without a verdict.
"""

import json
import os
import sys

import numpy as np

HELP, COVER = 0.90, (0.80, 0.95)


def load(folder):
    """(baselines dict or None, {size: {mode: [run, ...]}})"""
    base, runs = None, {}
    if not os.path.isdir(folder):
        return base, runs
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(folder, f)))
        if r["mode"] == "baselines":
            base = r
        else:
            runs.setdefault(r["size"], {}).setdefault(r["mode"], []).append(r)
    return base, runs


def stat(rs, key):
    v = np.array([r["test"][key] for r in rs], float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), len(v)


def verdict(base, modes):
    """[(check, passed, detail)] for one size, or None if a mode is missing."""
    if not base or "scratch" not in modes or "finetune" not in modes:
        return None
    ft, _, _ = stat(modes["finetune"], "crps")
    sc, _, _ = stat(modes["scratch"], "crps")
    cov, _, _ = stat(modes["finetune"], "cover90")
    out = [("Pretraining helps (fine-tuned <= 0.90 x scratch)", ft <= HELP * sc, f"{ft:.3f} vs {sc:.3f} "
            f"(ratio {ft / sc:.3f})"),
           ("Beats persistence", ft < base["persist_crps"], f"{ft:.3f} vs {base['persist_crps']:.3f}")]
    if "seasonal_crps" in base:
        out.append(("Beats seasonal-naive", ft < base["seasonal_crps"], f"{ft:.3f} vs {base['seasonal_crps']:.3f}"))
    out.append(("Calibrated (90% range holds truth 80-95%)", COVER[0] <= cov <= COVER[1], f"{cov:.2f}"))
    return out


def section(title, base, runs, with_verdict):
    lines = [f"## {title}", ""]
    if base:
        b = f"Persistence {base['persist_crps']:.3f}"
        b += f"; seasonal-naive {base['seasonal_crps']:.3f}" if "seasonal_crps" in base else ""
        lines += [f"Windows: {base['windows']}. Baselines (CRPS): {b}.", ""]
    lines += ["| Size | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds |",
              "|---|---|---|---|---|---|"]
    for size, modes in runs.items():
        cells, n = [], []
        for mode in ("scratch", "finetune"):
            if mode in modes:
                m, sd, k = stat(modes[mode], "crps")
                cells.append(f"{m:.3f} ± {sd:.3f}")
                n.append(f"{mode} {k}")
            else:
                cells.append("-")
        ratio = (f"{stat(modes['finetune'], 'crps')[0] / stat(modes['scratch'], 'crps')[0]:.3f}"
                 if {"scratch", "finetune"} <= set(modes) else "-")
        cov = f"{stat(modes['finetune'], 'cover90')[0]:.2f}" if "finetune" in modes else "-"
        lines.append(f"| {size} | {cells[0]} | {cells[1]} | {ratio} | {cov} | {', '.join(n)} |")
    if with_verdict:
        lines += ["", "### Verdict (criteria fixed before running)", ""]
        for size, modes in runs.items():
            v = verdict(base, modes)
            if v is None:
                lines.append(f"- **{size}: NOT YET** (needs baselines, scratch and fine-tuned runs)")
                continue
            ok = all(p for _, p, _ in v)
            lines.append(f"- **{size}: {'PASS' if ok else 'FAIL'}**")
            lines += [f"  - {c}: {'yes' if p else 'NO'} ({d})" for c, p, d in v]
    return lines + [""]


def report(root, dataset):
    if not os.path.isdir(root):
        sys.exit(f"No such results folder: {root}")
    lines = [f"# Phase 1: {dataset}", "",
             "Does pretraining on 8 simulated systems help forecast this real system? CRPS in normalised "
             "units, mean ± sd over seeds; lower is better.", ""]
    for regime, title, v in [("small", "Small-data regime (verdict)", True), ("full", "Full-data regime", False)]:
        base, runs = load(os.path.join(root, f"{dataset}_{regime}"))
        if base or runs:
            lines += section(title, base, runs, v)
    text = "\n".join(lines)
    out = os.path.join(root, f"{dataset}_PHASE1.md")
    open(out, "w").write(text)
    print(text)
    print(f"Written: {out}")
    return text


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    report(os.path.expanduser(sys.argv[1]), sys.argv[2])
