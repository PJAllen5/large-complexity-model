"""Compare two Phase 0 result folders (e.g. laptop vs ORCA).

    python orca/compare.py results results_orca
    python orca/compare.py --write-checksums results     # laptop only, once
    python orca/compare.py --noise results_orca ~/lcm-runs/phase0_seed1   # seed noise, no verdict

CRPS = continuous ranked probability score: how far the whole predicted range
of futures is from what happened (lower is better).

Verdicts:
  PASS                every CRPS within 10% and the same three headline answers
  PASS WITH WARNINGS  only noisy differences: few-shot runs (64 training
                      episodes) moved more than 10%, or a headline answer flipped
                      on a near-tie (the two scores were within 5% in either run)
  FAIL                a main run (joint, specialist, never-seen-it) moved more
                      than 10%, a headline answer flipped on a clear gap, runs
                      are missing, or the simulated data differs from the laptop's
                      by more than float rounding (see "data identity" below)

The 10% and 5% thresholds are judgement calls, not measured noise. Running the
same experiment twice with different seeds would measure the real spread.
"""

import hashlib
import json
import os
import sys

import numpy as np

TOL = 0.10        # max relative CRPS difference for a main run
CLOSE = 0.05      # comparisons closer than this count as near-ties
HELD_OUT = ["neurons", "epidemic", "kuramoto"]
NOISY_PREFIXES = ("finetune_", "scratch_")   # trained on only 64 episodes
CHECKSUMS = "data_checksums.json"


def load(folder):
    d = os.path.join(folder, "runs")
    if not os.path.isdir(d):
        sys.exit(f"No runs found in {d}")
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith(".json")}


# ---------------------------------------------------------------- data identity
# Simulated data is regenerated from fixed seeds on every machine. Discrete
# simulators reproduce bit for bit, but physics-style ones (boids, Lorenz96,
# n-body) pick up last-digit rounding differences between CPUs (seen: Apple M3
# vs ORCA's AMD EPYC, max 1.6e-4 of the data spread). So the check accepts data
# that is bit-identical OR whose per-episode summary (mean, std, min, max) is
# within DATA_TOL of the data spread, with identical masks and graphs. A
# different seed or a simulator bug changes those summaries by far more.
# Blind spot: a single changed value out of ~3,000 per episode goes unnoticed.
DATA_TOL = 1e-3
FINGERPRINT = "data_fingerprint.npz"


def _file_info(path):
    arrs = np.load(path)
    exact, structure = hashlib.sha256(), hashlib.sha256()
    for k in sorted(arrs.files):
        a = np.ascontiguousarray(arrs[k])
        for h in (exact,) if k == "x" else (exact, structure):
            h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
            h.update(a.tobytes())
    x = arrs["x"].astype(np.float64)
    stats = np.stack([x.mean((1, 2, 3)), x.std((1, 2, 3)), x.min((1, 2, 3)), x.max((1, 2, 3))], 1)
    return dict(exact=exact.hexdigest(), structure=structure.hexdigest(), spread=float(x.std())), stats


def data_info(folder):
    """{file: (info, per-episode stats)} for every simulated dataset in folder/data."""
    d = os.path.join(folder, "data")
    if not os.path.isdir(d):
        return {}
    return {f: _file_info(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".npz")}


def write_reference(folder):
    info = data_info(folder)
    if not info:
        sys.exit(f"No .npz files in {folder}/data")
    json.dump({f: i for f, (i, _) in info.items()}, open(os.path.join(folder, CHECKSUMS), "w"), indent=1)
    np.savez_compressed(os.path.join(folder, FINGERPRINT), **{f: st.astype(np.float32) for f, (_, st) in info.items()})
    print(f"wrote {folder}/{CHECKSUMS} and {folder}/{FINGERPRINT} for {len(info)} files")


def check_data(ref_dir, new_dir):
    """Returns (ok, message)."""
    path, fpath = os.path.join(ref_dir, CHECKSUMS), os.path.join(ref_dir, FINGERPRINT)
    if not (os.path.exists(path) and os.path.exists(fpath)):
        return True, f"data check skipped: no {path} / {fpath}"
    ref, ref_stats = json.load(open(path)), np.load(fpath)
    new = data_info(new_dir)
    if not new:
        return True, f"data check skipped: no {new_dir}/data folder"
    identical, close, bad = [], [], []
    for f, r in ref.items():
        if f not in new:
            bad.append(f"{f} (missing)")
            continue
        info, stats = new[f]
        if info["exact"] == r["exact"]:
            identical.append(f)
            continue
        if stats.shape != ref_stats[f].shape:
            bad.append(f"{f} (different size: {len(stats)} vs {len(ref_stats[f])} episodes)")
            continue
        if info["structure"] != r["structure"]:
            bad.append(f"{f} (masks/graphs differ)")
            continue
        worst = float(np.abs(stats - ref_stats[f]).max() / r["spread"])
        (close if worst <= DATA_TOL else bad).append(f"{f} ({worst:.1e} of spread)")
    if bad:
        return False, f"simulated data DIFFERS from the reference: {', '.join(bad)}"
    msg = f"simulated data matches ({len(identical)} files bit-identical"
    if close:
        msg += f"; {len(close)} within float rounding, worst: {max(close, key=lambda c: float(c.split('(')[1].split()[0]))}"
    return True, msg + ")"


# ---------------------------------------------------------------- headline answers
def headline_items(runs):
    """Each underlying comparison behind the three headline answers: (question, item, a, b), answer = a < b."""
    J = runs["joint"]["eval"]
    items = []
    for n in J:
        items.append(("joint beats specialist", n, J[n]["crps"], runs[f"specialist_{n}"]["eval"][n]["crps"]))
    for n in HELD_OUT:
        z = runs[f"loo_{n}"]["eval"][n]
        items.append(("zero-shot beats baseline", n, z["crps"], z["persist_crps"]))
    for n in HELD_OUT:
        items.append(("pretraining helps few-shot", n, runs[f"finetune_{n}"]["eval"][n]["crps"],
                      runs[f"scratch_{n}"]["eval"][n]["crps"]))
    return items


def _margin(a, b):
    return abs(a / b - 1) if b else float("inf")


def main(ref_dir, new_dir):
    ref, new = load(ref_dir), load(new_dir)
    common = sorted(set(ref) & set(new))
    missing = sorted(set(ref) - set(new))
    fails, warns = [], []

    ok, msg = check_data(ref_dir, new_dir)
    print(f"Data: {msg}\n")
    if not ok:
        fails.append("simulated data differs, so the runs aren't comparable")

    print(f"{'run':28s} {'system':14s} {'CRPS ref':>9s} {'CRPS new':>9s} {'diff':>7s}")
    n_ok = n_all = 0
    for tag in common:
        noisy = tag.startswith(NOISY_PREFIXES)
        for dom, m in ref[tag]["eval"].items():
            if dom not in new[tag]["eval"]:
                continue
            a, b = m["crps"], new[tag]["eval"][dom]["crps"]
            rel = b / a - 1
            n_all += 1
            note = ""
            if abs(rel) <= TOL:
                n_ok += 1
            elif noisy:
                note = "  <-- outside 10% (few-shot run: warning)"
                warns.append(f"{tag} {dom} moved {rel:+.0%}")
            else:
                note = "  <-- outside 10%"
                fails.append(f"{tag} {dom} moved {rel:+.0%}")
            print(f"{tag:28s} {dom:14s} {a:9.3f} {b:9.3f} {rel:+7.1%}{note}")
    print(f"\nScores within {TOL:.0%}: {n_ok} of {n_all}")

    if missing:
        print(f"Runs missing from {new_dir}: {', '.join(missing)}")
        fails.append(f"{len(missing)} runs missing")
    else:
        r_items, n_items = headline_items(ref), headline_items(new)
        counts = {}
        for (q, item, a1, b1), (_, _, a2, b2) in zip(r_items, n_items):
            c = counts.setdefault(q, [0, 0, 0])
            c[0] += a1 < b1
            c[1] += a2 < b2
            c[2] += 1
            if (a1 < b1) != (a2 < b2):
                if min(_margin(a1, b1), _margin(a2, b2)) < CLOSE:
                    warns.append(f"'{q}' flipped for {item} on a near-tie ({a1:.3f} vs {b1:.3f} -> {a2:.3f} vs {b2:.3f})")
                else:
                    fails.append(f"'{q}' flipped for {item} ({a1:.3f} vs {b1:.3f} -> {a2:.3f} vs {b2:.3f})")
        print(f"\n{'headline question':28s} {'ref':>8s} {'new':>8s}")
        for q, (cr, cn, tot) in counts.items():
            print(f"{q:28s} {f'{cr} of {tot}':>8s} {f'{cn} of {tot}':>8s}{'' if cr == cn else '  <-- differs'}")

    for w in warns:
        print(f"WARNING: {w}")
    for f in fails:
        print(f"FAILED: {f}")
    verdict = "FAIL" if fails else ("PASS WITH WARNINGS" if warns else "PASS")
    print(f"\nVERDICT: {verdict}")
    print("(thresholds are judgement calls: 10% per main-run score, 5% near-tie; see orca/compare.py)")
    return verdict


def noise(ref_dir, new_dir):
    """Measure run-to-run noise between two runs of the same experiment that differ
    only in training seed. No verdict: this is a measurement."""
    ref, new = load(ref_dir), load(new_dir)
    common = sorted(set(ref) & set(new))
    missing = sorted(set(ref) - set(new))

    laptop = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    print(f"Data vs laptop reference: {check_data(laptop, new_dir)[1]}\n")

    rows = []   # (group, tag, dom, rel crps diff, abs odds-error diff)
    for tag in common:
        group = "few-shot (64 episodes)" if tag.startswith(NOISY_PREFIXES) else "main runs"
        for dom, m in ref[tag]["eval"].items():
            if dom in new[tag]["eval"]:
                n = new[tag]["eval"][dom]
                rows.append((group, tag, dom, n["crps"] / m["crps"] - 1, n["prob_error"] - m["prob_error"]))

    print(f"{'run':28s} {'system':14s} {'CRPS diff':>10s} {'odds-error diff':>16s}")
    for g, tag, dom, rc, po in rows:
        print(f"{tag:28s} {dom:14s} {rc:+10.1%} {po:+16.3f}")

    print(f"\n{'group':24s} {'n':>3s} {'CRPS |diff| median':>19s} {'max':>7s} {'odds |diff| median':>19s} {'max':>7s}")
    for g in ("main runs", "few-shot (64 episodes)"):
        rc = np.array([abs(r[3]) for r in rows if r[0] == g])
        po = np.array([abs(r[4]) for r in rows if r[0] == g])
        if len(rc):
            print(f"{g:24s} {len(rc):3d} {np.median(rc):19.1%} {rc.max():7.1%} {np.median(po):19.3f} {po.max():7.3f}")
    top = sorted(rows, key=lambda r: -abs(r[3]))[:3]
    print("Largest CRPS moves: " + ", ".join(f"{t} {d} {rc:+.1%}" for _, t, d, rc, _ in top))

    if missing:
        print(f"\nRuns missing from {new_dir}: {', '.join(missing)}")
        return
    print(f"\n{'headline question':28s} {'seed A':>8s} {'seed B':>8s}")
    counts, flips = {}, []
    for (q, item, a1, b1), (_, _, a2, b2) in zip(headline_items(ref), headline_items(new)):
        c = counts.setdefault(q, [0, 0, 0])
        c[0] += a1 < b1
        c[1] += a2 < b2
        c[2] += 1
        if (a1 < b1) != (a2 < b2):
            flips.append(f"'{q}' for {item}: {a1:.3f} vs {b1:.3f} -> {a2:.3f} vs {b2:.3f}")
    for q, (ca, cb, tot) in counts.items():
        print(f"{q:28s} {f'{ca} of {tot}':>8s} {f'{cb} of {tot}':>8s}")
    print("Flipped answers: " + ("; ".join(flips) if flips else "none"))
    print("\nOne pair of seeds gives a rough noise estimate, not a precise one.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-checksums":
        write_reference(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "--noise":
        noise(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
