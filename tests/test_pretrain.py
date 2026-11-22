"""End-to-end check of pretrain.py on tiny synthetic Tycho data and tiny simulated data."""

import json
import os
import subprocess
import sys

import pytest

from lcm import tycho
from lcm.data import build_datasets

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _tiny_tycho(folder):
    rows = []
    for k in range(80):
        y, w = (1950, k + 1) if k < 52 else (1951, k - 51)
        for st in ("NJ", "NY", "PA", "IL", "IN", "MI"):
            rows.append([y * 100 + w, st, st, "STATE", "MEASLES", 5, 1.0 + (k % 7)])
            rows.append([y * 100 + w, st, st, "STATE", "POLIO", 2, 0.5 + (k % 5)])
    import csv
    with open(os.path.join(folder, tycho.CSV_NAME), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epi_week", "state", "loc", "loc_type", "disease", "cases", "incidence_per_100000"])
        wr.writerows(rows)


@pytest.mark.parametrize("variant", ["tycho", "simtycho"])
def test_pretrain_writes_checkpoint_and_record(tmp_path, monkeypatch, variant):
    monkeypatch.setattr(tycho, "MIN_WINDOWS", 10)
    ty, sims, out = tmp_path / "tycho", tmp_path / "sims", tmp_path / "out"
    ty.mkdir()
    _tiny_tycho(str(ty))
    tycho.build(str(ty))
    if variant == "simtycho":
        build_datasets(str(sims), n_train=8, n_test=2, n_val=4)
    r = subprocess.run([sys.executable, "pretrain.py", "--variant", variant, "--d", "32", "--layers", "1", "--heads", "2",
                        "--steps", "4", "--every", "2", "--tycho", str(ty), "--sim-data", str(sims), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "checkpoints" / "joint.pt").exists()
    info = json.load(open(out / "pretrain.json"))
    assert info["variant"] == variant and info["stopped_at"] == 4 and [h[0] for h in info["loss"]] == [2, 4]
    val_sets = set(info["val_by_system"][0][1])
    if variant == "tycho":
        assert val_sets == {"tycho_measles", "tycho_polio"} and set(info["weights"].values()) == {1.0}
    else:
        assert "tycho" in val_sets and "nbody" in val_sets and info["weights"]["tycho_polio"] == 0.5
    again = subprocess.run([sys.executable, "pretrain.py", "--variant", variant, "--out", str(out),
                            "--tycho", str(ty)], cwd=ROOT, capture_output=True, text=True)
    assert "skipping" in again.stdout                     # a finished run is not redone
