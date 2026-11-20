"""Compare two Phase 0 result folders (e.g. laptop vs ORCA).

    python orca/compare.py results results_orca
    python orca/compare.py --write-checksums results     # laptop only, once

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
def data_checksums(folder):
    """sha256 of the array contents of every simulated dataset in folder/data."""
    d = os.path.join(folder, "data")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            arrs = np.load(os.path.join(d, f))
            h = hashlib.sha256()
            for k in sorted(arrs.files):
                a = np.ascontiguousarray(arrs[k])
                h.update(f"{k}|{a.dtype}|{a.shape}|".encode())
                h.update(a.tobytes())
            out[f] = h.hexdigest()
    return out


def check_data(ref_dir, new_dir):
    """Returns (ok, message)."""
    path = os.path.join(ref_dir, CHECKSUMS)
    if not os.path.exists(path):
        return True, f"data check skipped: no {path}"
    ref = json.load(open(path))
    new = data_checksums(new_dir)
    if not new:
        return True, f"data check skipped: no {new_dir}/data folder"
    bad = sorted(f for f in ref if new.get(f) != ref[f])
    if bad:
        return False, f"simulated data DIFFERS from the reference for: {', '.join(bad)}"
    return True, f"simulated data identical to the reference ({len(ref)} files)"


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


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-checksums":
        sums = data_checksums(sys.argv[2])
        if not sums:
            sys.exit(f"No .npz files in {sys.argv[2]}/data")
        json.dump(sums, open(os.path.join(sys.argv[2], CHECKSUMS), "w"), indent=1)
        print(f"wrote {len(sums)} checksums to {sys.argv[2]}/{CHECKSUMS}")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
