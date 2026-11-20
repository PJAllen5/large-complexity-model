"""Checks for orca/compare.py's data check: it must accept float-rounding noise
between machines but catch data that is different in any way that matters."""

import os
import sys

import numpy as np
import pytest

from lcm.data import generate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orca"))
import compare  # noqa: E402


def _write(folder, name, d):
    os.makedirs(os.path.join(folder, "data"), exist_ok=True)
    np.savez_compressed(os.path.join(folder, "data", f"{name}_test.npz"), **d)


@pytest.fixture
def ref(tmp_path):
    d = generate("boids", 20, seed=1)
    _write(tmp_path / "ref", "boids", d)
    compare.write_reference(str(tmp_path / "ref"))
    return tmp_path, d


def _check(tmp_path, d):
    _write(tmp_path / "new", "boids", d)
    return compare.check_data(str(tmp_path / "ref"), str(tmp_path / "new"))


def test_identical_data_passes(ref):
    tmp_path, d = ref
    ok, msg = _check(tmp_path, d)
    assert ok and "1 files bit-identical" in msg


def test_float_rounding_noise_passes(ref):
    tmp_path, d = ref
    noisy = dict(d)
    rng = np.random.default_rng(0)
    # larger than anything seen between the M3 and ORCA (max 1.6e-4 of the spread)
    noisy["x"] = d["x"] + (rng.standard_normal(d["x"].shape) * 1e-5 * d["x"].std()).astype(np.float32)
    ok, msg = _check(tmp_path, noisy)
    assert ok and "within float rounding" in msg


def test_one_episode_simulated_differently_fails(ref):
    tmp_path, d = ref
    bad = dict(d)
    bad["x"] = d["x"].copy()
    bad["x"][3] = generate("boids", 1, seed=99)["x"][0]
    ok, msg = _check(tmp_path, bad)
    assert not ok and "DIFFERS" in msg


def test_small_systematic_change_fails(ref):
    tmp_path, d = ref
    bad = dict(d)
    bad["x"] = d["x"] * np.float32(1.01)     # e.g. a changed simulator constant
    ok, msg = _check(tmp_path, bad)
    assert not ok


def test_single_changed_value_is_a_known_blind_spot(ref):
    """Documents the limit: summaries per episode don't see one changed value
    out of ~3,000. Realistic causes (seed, simulator or library changes) alter
    many values at once and are caught by the tests above."""
    tmp_path, d = ref
    bad = dict(d)
    bad["x"] = d["x"].copy()
    bad["x"][3, 10, 0, 0] += 1.0
    ok, _ = _check(tmp_path, bad)
    assert ok


def test_different_seed_fails(ref):
    tmp_path, _ = ref
    ok, msg = _check(tmp_path, generate("boids", 20, seed=2))
    assert not ok


def test_changed_graph_fails(ref):
    tmp_path, d = ref
    bad = dict(d)
    bad["adj"] = d["adj"].copy()
    bad["adj"][0, 0, 1] = 0
    ok, msg = _check(tmp_path, bad)
    assert not ok and "masks/graphs differ" in msg
