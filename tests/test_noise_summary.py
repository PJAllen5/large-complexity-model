"""Checks for orca/noise_summary.py on synthetic seed folders with known answers."""

import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orca"))
import noise_summary as ns  # noqa: E402

BASE = os.path.join(os.path.dirname(__file__), "..", "results_orca")


def _seed_folder(tmp_path, seed, scale=None):
    """Copy the ORCA seed-0 runs, mark them with `seed`, and scale chosen CRPS values."""
    d = tmp_path / f"seed{seed}"
    shutil.copytree(os.path.join(BASE, "runs"), d / "runs")
    for f in os.listdir(d / "runs"):
        p = d / "runs" / f
        r = json.load(open(p))
        r["seed"] = seed
        for (tag, dom), k in (scale or {}).items():
            if r["tag"] == tag:
                r["eval"][dom]["crps"] *= k
        json.dump(r, open(p, "w"))
    return str(d)


def test_spread_is_computed_per_run(tmp_path):
    folders = [_seed_folder(tmp_path, s, {("joint", "neurons"): k}) for s, k in [(1, 0.9), (2, 1.0), (3, 1.1)]]
    out = ns.summarize(folders)
    vals = next(v for g, t, d, v in out["per_run"] if t == "joint" and d == "neurons")
    base = vals[1]
    assert np.allclose(vals, [0.9 * base, base, 1.1 * base])
    unchanged = next(v for g, t, d, v in out["per_run"] if t == "specialist_boids")
    assert unchanged.std() == 0


def test_seed_dependent_finding_is_counted(tmp_path):
    # make the joint model beat the epidemic specialist in one seed only
    folders = [_seed_folder(tmp_path, 1), _seed_folder(tmp_path, 2, {("joint", "epidemic"): 0.5})]
    out = ns.summarize(folders)
    assert out["findings"][("joint beats specialist", "epidemic")] == [False, True]
    assert out["findings"][("joint beats specialist", "kuramoto")] == [False, False]


def test_few_shot_advantage_ratio(tmp_path):
    folders = [_seed_folder(tmp_path, 1), _seed_folder(tmp_path, 2, {("scratch_epidemic", "epidemic"): 0.5})]
    adv = ns.summarize(folders)["advantage"]["epidemic"]
    assert adv[1] == pytest.approx(adv[0] * 0.5)


def test_duplicate_seeds_are_rejected(tmp_path):
    a = _seed_folder(tmp_path, 1)
    b = str(tmp_path / "copy")
    shutil.copytree(a, b)
    with pytest.raises(ValueError, match="duplicate seeds"):
        ns.summarize([a, b])


def test_seeds_needed_rule():
    assert ns.seeds_needed(0.10, 0.20) == 2      # 8 * 0.01 / 0.04 = 2
    assert ns.seeds_needed(0.20, 0.10) == 32     # 8 * 0.04 / 0.01 = 32
    assert ns.seeds_needed(0.0, 0.10) == 2       # never fewer than 2
