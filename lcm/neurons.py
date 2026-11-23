"""Allen Brain Observatory: real neuron activity (Phase 1, second dataset).

Source: Allen Brain Observatory - Visual Coding, 2-photon calcium imaging of mouse
visual cortex (Allen Institute), from the public AWS bucket
s3://allen-brain-observatory. Allen Institute Terms of Use: research and other
non-commercial use, with citation; no data goes in this repo. See
data/real/neurons_CARD.md.

    python -m lcm.neurons fetch ~/lcm-runs/neurons    # choose recordings, download, extract, delete (login node)

An episode is 12 neurons from one recording: their ΔF/F (fluorescence change
relative to baseline, a proxy for activity) as the one feature, averaged from
30 Hz down to 7.5 Hz, so a 64-step window is 8.5 s (6.4 s history, 2.1 s
forecast). Every pair of the 12 is linked (their wiring isn't known).

Design (decided 2026-11-22, before any data was used; PROJECT_LOG.md):
  - session type A only (the same stimuli in every recording), one recording per mouse;
  - recordings with fewer than 12 neurons are skipped; 12 neurons picked at random
    (seeded by the recording id);
  - split by mouse: 30 train, 5 validation, 10 test recordings, in a fixed shuffled order;
  - training windows every 16 steps; validation and test windows every 256 steps;
  - small-data regime: 25 evenly spaced windows from each of the first 4 training recordings.
"""

import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

from .data import T_TOTAL

BASE = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-2p/"
MANIFEST = "ophys_experiments.json"
DFF = "processing/brain_observatory_pipeline/DfOverF/imaging_plane_1/data"
SESSION = "three_session_A"
N_CELLS = 12
BIN = 4                          # 30 Hz frames averaged in fours -> 7.5 Hz
COUNTS = {"train": 30, "val": 5, "test": 10}
SEED = 0
STRIDE = {"train": 16, "val": 256, "test": 256}
SMALL_RECORDINGS, SMALL_PER_RECORDING = 4, 25
SUMMARY = "neurons_summary.json"


def download(url, path):
    """Stream `url` to `path`; checks the byte count against Content-Length."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        want = int(r.headers.get("Content-Length", -1))
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
    if want >= 0 and got != want:
        os.remove(tmp)
        raise IOError(f"{url}: got {got} bytes, expected {want}")
    os.replace(tmp, path)


def candidates(manifest):
    """Session-A experiment ids, one per mouse, in a fixed shuffled order."""
    exps = sorted((e for e in manifest if e["stimulus_name"] == SESSION), key=lambda e: e["id"])
    order = np.random.default_rng(SEED).permutation(len(exps))
    seen, out = set(), []
    for i in order:
        e = exps[i]
        if e["specimen_id"] not in seen:
            seen.add(e["specimen_id"])
            out.append(e)
    return out


def extract(nwb_path, exp_id):
    """(traces [T, 12] at 7.5 Hz, chosen cell indices, number of cells) or (None, None, n) if too few cells."""
    import h5py
    with h5py.File(nwb_path, "r") as f:
        dff = f[DFF][:]                                     # [cells, frames]
    n = dff.shape[0]
    if n < N_CELLS:
        return None, None, n
    cells = np.sort(np.random.default_rng(int(exp_id)).choice(n, N_CELLS, replace=False))
    x = dff[cells]
    t = x.shape[1] // BIN * BIN
    return x[:, :t].reshape(N_CELLS, -1, BIN).mean(-1).T.astype(np.float32), cells, n


def fetch(folder, fetch_file=download):
    """Choose the recordings, download each NWB file, keep 12 neurons' traces, delete the file.

    Resumable: progress is saved to neurons_summary.json after every recording."""
    os.makedirs(folder, exist_ok=True)
    man = os.path.join(folder, MANIFEST)
    if not os.path.exists(man):
        fetch_file(BASE + MANIFEST, man)
    spath = os.path.join(folder, SUMMARY)
    s = json.load(open(spath)) if os.path.exists(spath) else {
        "design": dict(session=SESSION, cells=N_CELLS, bin=BIN, counts=COUNTS, seed=SEED),
        **{k: [] for k in COUNTS}, "skipped": {}}
    done = {i for k in COUNTS for i in s[k]} | {int(i) for i in s["skipped"]}
    order = ["train", "val", "test"]
    for e in candidates(json.load(open(man))):
        need = [k for k in order if len(s[k]) < COUNTS[k]]
        if not need:
            break
        if e["id"] in done:
            continue
        out = os.path.join(folder, f"neurons_{e['id']}.npz")
        if not os.path.exists(out):
            nwb = os.path.join(folder, f"{e['id']}.nwb")
            try:
                fetch_file(f"{BASE}ophys_experiment_data/{e['id']}.nwb", nwb)
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    raise
                s["skipped"][str(e["id"])] = "not in the bucket"
                print(f"  {e['id']}: skipped (not in the bucket)")
                json.dump(s, open(spath, "w"), indent=1)
                continue
            try:
                x, cells, n = extract(nwb, e["id"])
            finally:
                os.remove(nwb)
            if x is None:
                s["skipped"][str(e["id"])] = f"{n} cells"
                print(f"  {e['id']}: skipped ({n} cells)")
                json.dump(s, open(spath, "w"), indent=1)
                continue
            np.savez_compressed(out, x=x, cells=cells, n_cells=n, specimen=e["specimen_id"],
                                area=e["targeted_structure"]["acronym"], depth=e["imaging_depth"])
        s[need[0]].append(e["id"])
        print(f"  {e['id']}: {need[0]} ({len(s[need[0]])}/{COUNTS[need[0]]})")
        json.dump(s, open(spath, "w"), indent=1)
    missing = {k: COUNTS[k] - len(s[k]) for k in COUNTS if len(s[k]) < COUNTS[k]}
    if missing:
        sys.exit(f"not enough usable recordings; still missing {missing}")
    print(f"done: {sum(len(s[k]) for k in COUNTS)} recordings, {len(s['skipped'])} skipped")
    return s


def episodes(folder, split, small=False):
    """(universal-format episodes, recording id of each window) for one split."""
    from .real import to_episodes
    s = json.load(open(os.path.join(folder, SUMMARY)))
    ids = s[split][:SMALL_RECORDINGS] if small else s[split]
    adj = 1 - np.eye(N_CELLS, dtype=np.float32)
    parts, groups = [], []
    for i in ids:
        with np.load(os.path.join(folder, f"neurons_{i}.npz")) as f:
            x = f["x"][:, :, None]
        T = len(x)
        if small:
            starts = np.linspace(0, T - T_TOTAL, SMALL_PER_RECORDING).round().astype(int).tolist()
        else:
            starts = list(range(0, T - T_TOTAL + 1, STRIDE[split]))
        d, kept = to_episodes(x, adj, starts)
        parts.append(d)
        groups += [int(i)] * len(kept)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, groups


def load(folder=None):
    """The Phase 1 dataset: prebuilt episodes per split (no single timeline, no period)."""
    folder = os.path.expanduser(folder or os.environ.get("LCM_NEURONS", "~/lcm-runs/neurons"))
    if not os.path.exists(os.path.join(folder, SUMMARY)):
        sys.exit(f"no neuron data in {folder}; run: python -m lcm.neurons fetch {folder}")
    eps = {"train": episodes(folder, "train"), "small_train": episodes(folder, "train", small=True),
           "val": episodes(folder, "val"), "test": episodes(folder, "test")}
    return dict(name="neurons", episodes=eps, period=None, units="ΔF/F (fluorescence change over baseline)")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "fetch":
        sys.exit(__doc__)
    fetch(os.path.expanduser(sys.argv[2]))
