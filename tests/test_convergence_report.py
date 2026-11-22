"""Checks for orca/convergence_report.py on synthetic folders with known gaps."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orca"))
import convergence_report as cr  # noqa: E402


def _runs(folder, size, seed, runs):
    d = os.path.join(folder, f"{size}_seed{seed}", "runs")
    os.makedirs(d, exist_ok=True)
    for tag, (ev, loss) in runs.items():
        json.dump({"tag": tag, "eval": ev, "loss": loss}, open(os.path.join(d, f"{tag}.json"), "w"))


def _m(c):
    return {"crps": c, "persist_crps": 1.0, "prob_error": 0.2}


IMPROVING = [[1000, 0.0, -2.0], [2000, 0.0, -2.5]]
OVERFIT = [[1000, 0.0, -2.0], [2000, 0.0, -2.5], [3000, 0.0, -2.3]]


def _setup(tmp_path, joint_gap, spec_crps=0.1, short_crps=0.12, sizes=("1.08M", "8.5M"), seeds=(0, 1, 2),
           overfit=False):
    joint, spec, short = (str(tmp_path / n) for n in ("joint", "spec", "short"))
    for size in sizes:
        for seed in seeds:
            _runs(joint, size, seed, {"joint": ({s: _m(spec_crps * joint_gap[size]) for s in cr.SYSTEMS}, [])})
            _runs(spec, size, seed, {f"specialist_{s}": ({s: _m(spec_crps)}, OVERFIT if overfit else IMPROVING)
                                     for s in cr.SYSTEMS})
            _runs(short, size, seed, {f"specialist_{s}": ({s: _m(short_crps)}, []) for s in cr.SYSTEMS})
    return joint, spec, short


@pytest.mark.parametrize("g, expected", [(0.92, "JOINT STILL WINS"), (0.97, "JOINT STILL WINS"),
                                         (1.0, "TIE"), (1.03, "TIE"), (1.04, "SPECIALISTS WIN")])
def test_verdict_thresholds(g, expected):
    assert cr.verdict(g) == expected


def test_gap_uses_the_long_specialists(tmp_path):
    joint, spec, short = _setup(tmp_path, {"1.08M": 1.0, "8.5M": 0.95})
    s = cr.report(joint, spec, short)
    assert s["8.5M"]["gaps"][0] == pytest.approx(0.95)
    text = open(os.path.join(spec, "CONVERGENCE.md")).read()
    assert "| 8.5M | 3 of 3 | 0.950 |" in text and "**JOINT STILL WINS**" in text and "**TIE**" in text
    assert "0.792 |" in text          # same joint against the 20k specialists: 0.95 x 0.1 / 0.12
    assert "| boids | 0.83 | 0.83 |" in text   # specialists gained 0.1 / 0.12 from 8x training


def test_sizes_without_long_specialists_are_left_out(tmp_path):
    joint, spec, _ = _setup(tmp_path, {"1.08M": 1.0, "8.5M": 1.0})
    _runs(joint, "28M", 0, {"joint": ({s: _m(0.1) for s in cr.SYSTEMS}, [])})
    assert "28M" not in cr.report(joint, spec)


def test_unfinished_seed_is_not_counted(tmp_path):
    joint, spec, _ = _setup(tmp_path, {"1.08M": 1.0, "8.5M": 1.1})
    os.remove(os.path.join(spec, "8.5M_seed2", "runs", "specialist_nbody.json"))
    s = cr.report(joint, spec)
    assert sorted(s["8.5M"]["gaps"]) == [0, 1]
    assert "| 8.5M | 2 of 3 |" in open(os.path.join(spec, "CONVERGENCE.md")).read()


@pytest.mark.parametrize("overfit, expected", [(True, (8, 0, 8)), (False, (0, 8, 8))])
def test_specialist_curves_are_counted(tmp_path, overfit, expected):
    _, spec, _ = _setup(tmp_path, {"1.08M": 1.0, "8.5M": 1.0}, overfit=overfit, seeds=(0,))
    assert cr.curve_states(cr.load_root(spec)["8.5M"]) == expected   # OVERFIT rises 0.2 after its best


BASE = {s: 1.0 for s in cr.SYSTEMS} | {"nbody": 1.36, "epidemic": 1.11}


@pytest.mark.parametrize("change, expected", [
    ({"nbody": 1.10, "epidemic": 1.03}, "SUPPORTED"),
    ({"nbody": 1.10, "epidemic": 1.03, "boids": 1.04}, "MIXED"),       # another system got worse
    ({"nbody": 1.33, "epidemic": 1.08}, "REJECTED"),                   # both moved < 0.05
    ({"nbody": 1.20, "epidemic": 1.10}, "MIXED"),                      # n-body moved, not enough
])
def test_hypothesis_check(change, expected):
    assert cr.hypothesis_check(BASE | change, BASE)[0] == expected


def test_report_with_compare_joint_and_val_peaks(tmp_path):
    joint, spec, _ = _setup(tmp_path, {"1.08M": 1.0, "8.5M": 1.0}, seeds=(0,))
    base = str(tmp_path / "base")
    log = [[1000, {s: -1.0 for s in cr.SYSTEMS}], [2000, {s: -2.0 if s == "nbody" else -0.5 for s in cr.SYSTEMS}]]
    for size in ("1.08M", "8.5M"):
        _runs(base, size, 0, {"joint": ({s: _m(0.1 * BASE[s]) for s in cr.SYSTEMS}, [])})
        path = os.path.join(joint, f"{size}_seed0", "runs", "joint.json")
        run = json.load(open(path))
        run["val_by_system"] = log
        json.dump(run, open(path, "w"))
    out = str(tmp_path / "out")
    os.makedirs(out)
    cr.report(joint, spec, compare_joint=base, out=out)
    text = open(os.path.join(out, "CONVERGENCE.md")).read()
    assert not os.path.exists(os.path.join(spec, "CONVERGENCE.md"))      # --out respected
    assert "| 8.5M | **SUPPORTED** | nbody 1.36 -> 1.00, epidemic 1.11 -> 1.00 |" in text
    assert "| nbody | 2k | 2k |" in text and "| boids | 1k | 1k |" in text
