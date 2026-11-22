"""Checks for Phase 1: real-data windows, baselines, the flu loader and the report verdict."""

import json
import os
import sys

import numpy as np
import pytest
import torch

from lcm.data import T_CTX, T_TOTAL
from lcm.real import load_flu, point_crps, seasonal_episodes, split_starts, to_episodes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phase1_report as pr  # noqa: E402


def _series(T=200, N=3, F=2, seed=0):
    return np.random.default_rng(seed).standard_normal((T, N, F)).astype(np.float32), np.eye(3, dtype=np.float32)


def test_windows_fit_inside_the_split():
    assert split_starts(10, 10 + T_TOTAL) == [10]
    assert split_starts(10, 10 + T_TOTAL - 1) == []
    assert split_starts(0, 100, stride=10) == [0, 10, 20, 30]


def test_windows_with_gaps_are_dropped_and_padded():
    v, adj = _series()
    v[30, 1, 0] = np.nan
    d, kept = to_episodes(v, adj, [0, 10, 100])
    assert kept == [100]                                   # 0 and 10 both contain step 30
    assert d["x"].shape == (1, T_TOTAL, 12, 4) and np.array_equal(d["x"][0, :, :3, :2], v[100:164])
    assert d["ent_mask"][0].tolist() == [1] * 3 + [0] * 9 and d["feat_mask"][0].tolist() == [1, 1, 0, 0]


def test_too_many_entities_is_refused():
    with pytest.raises(ValueError):
        to_episodes(np.zeros((70, 13, 1), np.float32), np.zeros((13, 13)), [0])


def test_seasonal_episode_uses_values_one_period_earlier():
    v, adj = _series()
    d, kept = seasonal_episodes(v, adj, [0, 60, 100], period=52)
    assert kept == [60, 100]                               # start 0 has nothing a period earlier
    i = kept.index(100)
    assert np.array_equal(d["x"][i, :T_CTX, :3, :2], v[100:100 + T_CTX])                    # real history
    assert np.array_equal(d["x"][i, T_CTX:, :3, :2], v[100 + T_CTX - 52:100 + T_TOTAL - 52])  # lagged forecast
    # overlapping windows must not change each other's history
    assert np.array_equal(d["x"][kept.index(60), :T_CTX, :3, :2], v[60:60 + T_CTX])


def test_point_crps():
    v, adj = _series()
    v[:, :, :] = np.linspace(0, 1, 200)[:, None, None]     # a steady rise
    d = {k: torch.from_numpy(x) for k, x in to_episodes(v, adj, [0])[0].items()}
    assert point_crps(d, d["x"].clone()) == pytest.approx(0.0, abs=1e-6)   # perfect forecast
    assert point_crps(d) > 0.1                                             # persistence misses the rise
    bad = d["x"].clone()
    bad[:, 0] += 1
    with pytest.raises(AssertionError):
        point_crps(d, bad)                                 # a forecast must share the history


def test_flu_loader():
    ds = load_flu()
    v, s = ds["values"], ds["splits"]
    assert v.shape[1:] == (10, 1) and np.isfinite(v).all() and ds["times"][0] == 200240
    assert (ds["adj"] == ds["adj"].T).all() and ds["adj"].diagonal().sum() == 0 and ds["adj"].sum() == 32
    assert s["train"][1] == s["val"][0] and s["val"][1] == s["test"][0]       # back to back, no overlap
    assert s["small_train"][1] == s["train"][1] and s["small_train"][0] > s["train"][0]
    t = ds["times"]
    assert t[s["val"][0]] == 201340 and t[s["test"][0]] == 201540 and t[s["small_train"][0]] == 201040
    assert len(split_starts(*s["small_train"])) == 93


def _run(folder, mode, size, seed, crps, cover=0.9):
    os.makedirs(folder, exist_ok=True)
    json.dump(dict(mode=mode, size=size, seed=seed, test=dict(crps=crps, cover90=cover)),
              open(os.path.join(folder, f"{mode}_{size}_seed{seed}.json"), "w"))


def _base(folder, persist=0.6, seasonal=0.5):
    os.makedirs(folder, exist_ok=True)
    json.dump(dict(mode="baselines", windows={}, persist_crps=persist, seasonal_crps=seasonal),
              open(os.path.join(folder, "baselines.json"), "w"))


@pytest.mark.parametrize("ft, cover, expected", [(0.40, 0.9, "PASS"),     # all four hold
                                                 (0.46, 0.9, "FAIL"),     # only 8% better than scratch
                                                 (0.40, 0.97, "FAIL"),    # over-wide ranges
                                                 (0.44, 0.9, "PASS")])
def test_report_verdict(tmp_path, ft, cover, expected):
    small = str(tmp_path / "flu_small")
    _base(small)
    for seed in range(3):
        _run(small, "scratch", "8.5M", seed, 0.50)
        _run(small, "finetune", "8.5M", seed, ft, cover)
    text = pr.report(str(tmp_path), "flu")
    assert f"**8.5M: {expected}**" in text


def test_report_beats_seasonal_check(tmp_path):
    small = str(tmp_path / "flu_small")
    _base(small, seasonal=0.30)                            # seasonal-naive is hard to beat
    for seed in range(2):
        _run(small, "scratch", "8.5M", seed, 0.50)
        _run(small, "finetune", "8.5M", seed, 0.40)
    text = pr.report(str(tmp_path), "flu")
    assert "**8.5M: FAIL**" in text and "Beats seasonal-naive: NO" in text
    assert "| 8.5M | 0.500 ± 0.000 | 0.400 ± 0.000 | 0.800 |" in text
