"""Project Tycho: real US epidemics of other diseases, for pretraining (Phase 1).

Source: Project Tycho Level 1 v1.0.0 (University of Pittsburgh), Zenodo record
12608992, CC BY. Cite: van Panhuis et al., "Contagious Diseases in the United
States from 1888 to the present", NEJM 2013; 369(22): 2152-2158. We modify the
data (select states and weeks, cut into windows); see data/real/tycho_CARD.md.

    python -m lcm.tycho fetch ~/lcm-runs/tycho        # download + checksum (login node)
    python -m lcm.tycho build ~/lcm-runs/tycho        # windows -> tycho_<disease>_{train,val}.npz

An episode is one disease in one US census division: the division's states that
reported **every** week of the 64-week window (at least 3, at most 9), as
entities; weekly incidence per 100,000 as the one feature; every pair of those
states linked. Weeks without a row are treated as unreported (Tycho records
zero-case weeks explicitly). Windows with almost no cases are dropped.

Validation: two whole census divisions (Mid-Atlantic and Pacific), 16-28% of each
disease's windows. Usable windows bunch up in a few years per disease, so a split
by calendar blocks was badly uneven (PROJECT_LOG.md 2026-11-22). Validation only
picks the pretraining checkpoint; it shares years (national epidemics) with
training, so it is somewhat optimistic. Diseases with fewer than MIN_WINDOWS
windows are left out (rubella: 14).
"""

import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np

from .data import FMAX, NMAX, T_TOTAL

URL = "https://zenodo.org/records/12608992/files/ProjectTycho_Level1_v1.0.0.zip?download=1"
MD5 = "4c3991594d8dbad117e69ef49808eb5b"
CSV_NAME = "ProjectTycho_Level1_v1.0.0.csv"
DIVISIONS = {   # US census divisions (at most 9 states each, so they fit the model's 12 entities)
    "new_england": "CT ME MA NH RI VT", "mid_atlantic": "NJ NY PA", "east_north_central": "IL IN MI OH WI",
    "west_north_central": "IA KS MN MO NE ND SD", "south_atlantic": "DE DC FL GA MD NC SC VA WV",
    "east_south_central": "AL KY MS TN", "west_south_central": "AR LA OK TX",
    "mountain": "AZ CO ID MT NV NM UT WY", "pacific": "AK CA HI OR WA"}
DIVISIONS = {k: v.split() for k, v in DIVISIONS.items()}
MIN_STATES = 3
MIN_ACTIVE = 0.25        # at least this share of (state, week) values above zero
VAL_DIVISIONS = ("mid_atlantic", "pacific")
MIN_WINDOWS = 100


def week_index(epiweek):
    """MMWR epi week (yyyyww) -> consecutive week number. Week 1 contains January 4; weeks start Sunday."""
    y, w = divmod(int(epiweek), 100)
    jan4 = dt.date(y, 1, 4)
    week1 = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
    return (week1 + dt.timedelta(weeks=w - 1) - dt.date(1900, 1, 7)).days // 7


def index_year(i):
    return (dt.date(1900, 1, 7) + dt.timedelta(weeks=int(i))).year


def fetch(folder):
    """Download the Zenodo zip, check its MD5, and extract the CSV into `folder`."""
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, CSV_NAME)
    if os.path.exists(out):
        print(f"{out} exists")
        return out
    raw = urllib.request.urlopen(URL, timeout=120).read()
    got = hashlib.md5(raw).hexdigest()
    if got != MD5:
        sys.exit(f"checksum mismatch: got {got}, expected {MD5}")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        data = z.read(CSV_NAME)
    tmp = f"{out}.{os.getpid()}.tmp"
    open(tmp, "wb").write(data)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(data) / 1e6:.1f} MB, checksum OK)")
    return out


def read(path):
    """{disease: {state: {week_index: incidence per 100k}}} for state-level rows."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["loc_type"] != "STATE":
                continue
            out.setdefault(r["disease"], {}).setdefault(r["state"], {})[week_index(r["epi_week"])] = \
                float(r["incidence_per_100000"])
    return out


def windows(series, states, stride=1):
    """Windows for one disease in one division: [(start, [states fully reported], values [64, n])]."""
    have = [s for s in states if s in series]
    if len(have) < MIN_STATES:
        return []
    lo = min(min(series[s]) for s in have)
    hi = max(max(series[s]) for s in have)
    out = []
    for start in range(lo, hi - T_TOTAL + 2, stride):
        weeks = range(start, start + T_TOTAL)
        full = [s for s in have if all(w in series[s] for w in weeks)]
        if len(full) < MIN_STATES:
            continue
        v = np.array([[series[s][w] for s in full] for w in weeks], np.float32)
        if (v > 0).mean() < MIN_ACTIVE:
            continue
        out.append((start, full, v))
    return out


def to_universal(wins):
    n = len(wins)
    x = np.zeros((n, T_TOTAL, NMAX, FMAX), np.float32)
    em = np.zeros((n, NMAX), np.float32)
    a = np.zeros((n, NMAX, NMAX), np.float32)
    for i, (_, full, v) in enumerate(wins):
        k = len(full)
        x[i, :, :k, 0] = v
        em[i, :k] = 1
        a[i, :k, :k] = 1 - np.eye(k)
    fm = np.zeros((n, FMAX), np.float32)
    fm[:, 0] = 1
    return dict(x=x, ent_mask=em, feat_mask=fm, adj=a)


def build(folder, stride=1):
    """Write tycho_<disease>_{train,val}.npz and tycho_summary.json into `folder`."""
    data = read(os.path.join(folder, CSV_NAME))
    summary = {}
    for disease, series in sorted(data.items()):
        parts = {"train": [], "val": []}
        for div, states in DIVISIONS.items():
            parts["val" if div in VAL_DIVISIONS else "train"] += windows(series, states, stride)
        name = disease.lower().replace(" ", "_")
        summary[name] = {s: len(v) for s, v in parts.items()}
        if sum(summary[name].values()) < MIN_WINDOWS:
            summary[name]["left_out"] = f"fewer than {MIN_WINDOWS} windows"
            print(f"  {name:12s} {summary[name]}")
            continue
        for s, wins in parts.items():
            if wins:
                np.savez_compressed(os.path.join(folder, f"tycho_{name}_{s}.npz"), **to_universal(wins))
        years = [index_year(w[0]) for v in parts.values() for w in v]
        summary[name]["years"] = [min(years), max(years)] if years else None
        print(f"  {name:12s} {summary[name]}")
    json.dump(summary, open(os.path.join(folder, "tycho_summary.json"), "w"), indent=1)
    return summary


def load(folder, split, device="cpu"):
    """{disease: tensors} for every disease built into `folder` (as listed in tycho_summary.json)."""
    import torch
    summary = json.load(open(os.path.join(folder, "tycho_summary.json")))
    out = {}
    for name, info in summary.items():
        path = os.path.join(folder, f"tycho_{name}_{split}.npz")
        if "left_out" not in info and os.path.exists(path):
            with np.load(path) as d:
                out[f"tycho_{name}"] = {k: torch.from_numpy(d[k]).to(device) for k in d.files}
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fetch", "build"):
        sys.exit(__doc__)
    folder = os.path.expanduser(sys.argv[2])
    fetch(folder) if sys.argv[1] == "fetch" else build(folder)
