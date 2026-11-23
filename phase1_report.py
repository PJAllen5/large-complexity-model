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
  beats history mean  fine-tuned CRPS below each entity's mean over its history (if it has none; neurons)
  calibrated          fine-tuned 90% range holds the truth 80-95% of the time
The full-data regime is reported without a verdict.

Also shown when the runs recorded them (the rerun with dense validation): the
pretrained model's score before any fine-tuning ("no fine-tuning"), the steps at
which validation was best, and 90% coverage by flu season of the forecast start.
"""

import json
import os
import sys

import numpy as np

HELP, COVER = 0.90, (0.80, 0.95)


def run_key(r):
    """scratch, finetune (simulator-pretrained), or finetune-<label> (other pretraining)."""
    if r["mode"] != "finetune" or r.get("pretrained", "sims") in (None, "sims"):
        return r["mode"]
    return f"finetune-{r['pretrained']}"


def load(folder):
    """(baselines dict or None, {size: {key: [run, ...]}}) with keys from run_key."""
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
            runs.setdefault(r["size"], {}).setdefault(run_key(r), []).append(r)
    return base, runs


def stat(rs, key):
    v = np.array([r["test"][key] for r in rs], float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), len(v)


def label(key):
    return "sims" if key == "finetune" else key[len("finetune-"):]


def verdict(base, modes, key="finetune"):
    """[(check, passed, detail)] for one size and pretraining, or None if a run type is missing.

    For pretraining other than the simulators (key finetune-<label>), one more check:
    it must beat the simulator-pretrained fine-tuned model (Tycho test, PROJECT_LOG 2026-11-22)."""
    if not base or "scratch" not in modes or key not in modes:
        return None
    ft, _, _ = stat(modes[key], "crps")
    sc, _, _ = stat(modes["scratch"], "crps")
    cov, _, _ = stat(modes[key], "cover90")
    out = [("Pretraining helps (fine-tuned <= 0.90 x scratch)", ft <= HELP * sc, f"{ft:.3f} vs {sc:.3f} "
            f"(ratio {ft / sc:.3f})"),
           ("Beats persistence", ft < base["persist_crps"], f"{ft:.3f} vs {base['persist_crps']:.3f}")]
    if "seasonal_crps" in base:
        out.append(("Beats seasonal-naive", ft < base["seasonal_crps"], f"{ft:.3f} vs {base['seasonal_crps']:.3f}"))
    if "mean_crps" in base:                # datasets with no period (neurons; PROJECT_LOG 2026-11-22)
        out.append(("Beats the history mean", ft < base["mean_crps"], f"{ft:.3f} vs {base['mean_crps']:.3f}"))
    out.append(("Calibrated (90% range holds truth 80-95%)", COVER[0] <= cov <= COVER[1], f"{cov:.2f}"))
    if key != "finetune":
        if "finetune" not in modes:
            return None
        sims = stat(modes["finetune"], "crps")[0]
        out.append(("Beats the simulator-pretrained fine-tuned model", ft < sims, f"{ft:.3f} vs {sims:.3f}"))
    return out


def season_of(epiweek):
    y, w = divmod(epiweek, 100)
    return y if w >= 40 else y - 1


def window_groups(r):
    """A label per test window: flu season of the forecast start, or the test recording (neurons)."""
    if "test_forecast_start" in r:
        return [season_of(wk) for wk in r["test_forecast_start"]]
    return r.get("test_group")


def coverage_by_season(runs):
    """{(size, mode): {season or recording: mean coverage over windows and seeds}} for runs with per-window data."""
    out = {}
    for size, modes in runs.items():
        for mode, rs in modes.items():
            acc = {}
            for r in rs:
                groups = window_groups(r)
                if "window_cover90" not in r["test"] or groups is None:
                    continue
                for g, c in zip(groups, r["test"]["window_cover90"]):
                    acc.setdefault(g, []).append(c)
            if acc:
                out[(size, mode)] = {k: float(np.mean(v)) for k, v in sorted(acc.items())}
    return out


def ft_keys(modes):
    """Fine-tuned run types present: the simulator one first, then the others alphabetically."""
    return sorted((k for k in modes if k.startswith("finetune")), key=lambda k: (k != "finetune", k))


def section(title, base, runs, with_verdict):
    lines = [f"## {title}", ""]
    if base:
        b = f"Persistence {base['persist_crps']:.3f}"
        b += f"; seasonal-naive {base['seasonal_crps']:.3f}" if "seasonal_crps" in base else ""
        b += f"; history mean {base['mean_crps']:.3f}" if "mean_crps" in base else ""
        lines += [f"Windows: {base['windows']}. Baselines (CRPS): {b}.", ""]
    lines += ["| Size | Pretraining | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) "
              "| Seeds | No fine-tuning | Coverage (scratch) | Best step scratch / fine-tuned |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for size, modes in runs.items():
        for key in ft_keys(modes) or ["finetune"]:
            cells, n = [], []
            for mode in ("scratch", key):
                if mode in modes:
                    m, sd, k = stat(modes[mode], "crps")
                    cells.append(f"{m:.3f} ± {sd:.3f}")
                    n.append(f"{'scratch' if mode == 'scratch' else 'finetune'} {k}")
                else:
                    cells.append("-")
            ratio = (f"{stat(modes[key], 'crps')[0] / stat(modes['scratch'], 'crps')[0]:.3f}"
                     if {"scratch", key} <= set(modes) else "-")
            cov = f"{stat(modes[key], 'cover90')[0]:.2f}" if key in modes else "-"
            s0 = [r["test_step0"]["crps"] for r in modes.get(key, []) if "test_step0" in r]
            zero = f"{np.mean(s0):.3f}" if s0 else "-"
            cov_sc = f"{stat(modes['scratch'], 'cover90')[0]:.2f}" if "scratch" in modes else "-"
            best = " / ".join(",".join(str(r.get("best_step", "?")) for r in modes.get(m, [])) or "-"
                              for m in ("scratch", key))
            lines.append(f"| {size} | {label(key)} | {cells[0]} | {cells[1]} | {ratio} | {cov} | {', '.join(n)} "
                         f"| {zero} | {cov_sc} | {best} |")
    seasons = coverage_by_season(runs)
    if seasons:
        cols = list(seasons)
        all_s = sorted({k for v in seasons.values() for k in v})
        by_season = not any("test_group" in r for modes in runs.values() for rs in modes.values() for r in rs)
        what, head = (("season of the forecast start", "Season") if by_season else ("test recording", "Recording"))
        lines += ["", f"90% coverage by {what} (mean over windows and seeds; target 0.90):", "",
                  f"| {head} | " + " | ".join(f"{sz} {m}" for sz, m in cols) + " |", "|---|" + "---|" * len(cols)]
        for g in all_s:
            name = f"{g}-{str(g + 1)[-2:]}" if by_season else str(g)
            lines.append(f"| {name} | "
                         + " | ".join(f"{seasons[c][g]:.2f}" if g in seasons[c] else "-" for c in cols) + " |")
    if with_verdict:
        lines += ["", "### Verdict (criteria fixed before running)", ""]
        for size, modes in runs.items():
            for key in ft_keys(modes) or ["finetune"]:
                v = verdict(base, modes, key)
                if v is None:
                    lines.append(f"- **{size} ({label(key)}): NOT YET** (needs baselines, scratch and "
                                 "fine-tuned runs)")
                    continue
                ok = all(p for _, p, _ in v)
                lines.append(f"- **{size} ({label(key)}): {'PASS' if ok else 'FAIL'}**")
                lines += [f"  - {c}: {'yes' if p else 'NO'} ({d})" for c, p, d in v]
    return lines + [""]


def report(root, dataset):
    if not os.path.isdir(root):
        sys.exit(f"No such results folder: {root}")
    lines = [f"# Phase 1: {dataset}", "",
             "Does pretraining help forecast this real system? CRPS in normalised "
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
