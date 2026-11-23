"""Checks for the Allen neuron loader (lcm/neurons.py) on synthetic NWB-like files, and the
report's history-mean check and per-recording coverage."""

import json
import os
import shutil
import sys

import h5py
import numpy as np
import pytest

from lcm import neurons
from lcm.data import T_TOTAL

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import phase1_report as pr  # noqa: E402


def _nwb(path, cells, frames, seed=0):
    dff = np.random.default_rng(seed).standard_normal((cells, frames)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f[neurons.DFF] = dff
    return dff


def test_extract_picks_12_cells_and_averages_frames(tmp_path):
    p = str(tmp_path / "1.nwb")
    dff = _nwb(p, cells=20, frames=4 * 50 + 3)               # 3 leftover frames are dropped
    x, cells, n = neurons.extract(p, 123)
    assert n == 20 and x.shape == (50, 12) and len(set(cells)) == 12
    assert np.allclose(x[7], dff[cells, 28:32].mean(1))      # step 7 = mean of frames 28-31
    assert np.array_equal(cells, neurons.extract(p, 123)[1])  # same recording id, same cells
    _nwb(str(tmp_path / "2.nwb"), cells=11, frames=40)
    assert neurons.extract(str(tmp_path / "2.nwb"), 5) == (None, None, 11)   # too few cells


def _manifest(n_mice=60, extra_per_mouse=1):
    out, i = [], 1000
    for m in range(n_mice):
        for k, sess in enumerate(["three_session_A"] * extra_per_mouse + ["three_session_B"]):
            out.append(dict(id=i, specimen_id=m, stimulus_name=sess, imaging_depth=175,
                            targeted_structure=dict(acronym="VISp")))
            i += 1
    return out


def test_candidates_are_session_a_one_per_mouse_and_fixed():
    c = neurons.candidates(_manifest(extra_per_mouse=2))
    assert all(e["stimulus_name"] == "three_session_A" for e in c)
    assert len({e["specimen_id"] for e in c}) == len(c) == 60
    assert [e["id"] for e in c] == [e["id"] for e in neurons.candidates(_manifest(extra_per_mouse=2))]


def _fake_fetch(small_ids, frames=4 * 700):
    """A stand-in for download(): writes the manifest, or a synthetic NWB file."""
    def fetch_file(url, path):
        if url.endswith(neurons.MANIFEST):
            json.dump(_manifest(), open(path, "w"))
        else:
            i = int(os.path.basename(path).split(".")[0])
            _nwb(path, cells=5 if i in small_ids else 15, frames=frames, seed=i)
    return fetch_file


def test_fetch_splits_by_mouse_skips_small_recordings_and_resumes(tmp_path):
    order = [e["id"] for e in neurons.candidates(_manifest())]
    small = set(order[1:3])
    s = neurons.fetch(str(tmp_path), _fake_fetch(small))
    chosen = s["train"] + s["val"] + s["test"]
    assert [len(s[k]) for k in ("train", "val", "test")] == [30, 5, 10]
    assert sorted(map(int, s["skipped"])) == sorted(small) and not small & set(chosen)
    assert chosen == [i for i in order if i not in small][:45]        # fixed order: train, then val, then test
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".nwb")]   # big files deleted
    calls = []
    s2 = neurons.fetch(str(tmp_path), lambda url, path: calls.append(url))
    assert s2 == s and calls == []                                    # resume: nothing downloaded again


def test_fetch_stops_if_too_few_usable_recordings(tmp_path):
    order = [e["id"] for e in neurons.candidates(_manifest())]
    with pytest.raises(SystemExit, match="missing"):
        neurons.fetch(str(tmp_path), _fake_fetch(set(order[:20])))


def test_episodes_strides_small_regime_and_load(tmp_path, monkeypatch):
    neurons.fetch(str(tmp_path), _fake_fetch(set(), frames=4 * 700))
    T = 700
    d, g = neurons.episodes(str(tmp_path), "train")
    per = len(range(0, T - T_TOTAL + 1, 16))
    assert d["x"].shape == (30 * per, T_TOTAL, 12, 4) and len(g) == 30 * per
    assert d["ent_mask"].sum(1).tolist() == [12] * len(g) and d["adj"][0].trace() == 0
    small, gs = neurons.episodes(str(tmp_path), "train", small=True)
    assert len(gs) == 4 * 25 and sorted(set(gs)) == sorted(json.load(open(tmp_path / neurons.SUMMARY))["train"][:4])
    v, gv = neurons.episodes(str(tmp_path), "val")
    assert len(gv) == 5 * len(range(0, T - T_TOTAL + 1, 256))
    monkeypatch.setenv("LCM_NEURONS", str(tmp_path))
    ds = neurons.load()
    assert ds["period"] is None and set(ds["episodes"]) == {"train", "small_train", "val", "test"}
    test_ids = set(ds["episodes"]["test"][1])
    assert not test_ids & set(ds["episodes"]["train"][1]) and not test_ids & set(ds["episodes"]["val"][1])


def _run(folder, mode, seed, crps, groups, key=None):
    os.makedirs(folder, exist_ok=True)
    r = dict(mode=mode, size="8.5M", seed=seed, test=dict(crps=crps, cover90=0.85, window_cover90=[1.0, 0.7, 0.9]),
             test_group=groups)
    if key:
        r["pretrained"] = key
    json.dump(r, open(os.path.join(folder, f"{mode}{'-' + key if key else ''}_8.5M_seed{seed}.json"), "w"))


def test_report_history_mean_check_and_coverage_by_recording(tmp_path):
    small = str(tmp_path / "neurons_small")
    os.makedirs(small)
    json.dump(dict(mode="baselines", windows={}, persist_crps=0.6, mean_crps=0.42),
              open(os.path.join(small, "baselines.json"), "w"))
    for seed in range(2):
        _run(small, "scratch", seed, 0.50, [11, 11, 22])
        _run(small, "finetune", seed, 0.40, [11, 11, 22])
        _run(small, "finetune", seed, 0.44, [11, 11, 22], key="tycho")
    text = pr.report(str(tmp_path), "neurons")
    assert "history mean 0.420" in text and "seasonal" not in text
    assert "**8.5M (sims): PASS**" in text and "Beats the history mean: yes (0.400 vs 0.420)" in text
    assert "**8.5M (tycho): FAIL**" in text and "Beats the history mean: NO (0.440 vs 0.420)" in text
    assert "coverage by test recording" in text and "| 11 | 0.85 | 0.85 | 0.85 |" in text and "| 22 | 0.90 |" in text


@pytest.mark.skipif(not os.environ.get("LCM_NWB_SAMPLE"), reason="set LCM_NWB_SAMPLE to a real Allen NWB file")
def test_real_allen_file(tmp_path):
    """The real file layout: the DfOverF path exists and holds [cells, frames]."""
    x, cells, n = neurons.extract(os.environ["LCM_NWB_SAMPLE"], 1)
    assert n >= 1 and (x is None) == (n < 12)
    if x is not None:
        assert x.shape[1] == 12 and np.isfinite(x).all()


def test_fetch_skips_recordings_missing_from_the_bucket(tmp_path):
    import urllib.error
    order = [e["id"] for e in neurons.candidates(_manifest())]
    inner = _fake_fetch(set())

    def fetch_file(url, path):
        if url.endswith(f"/{order[0]}.nwb"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        inner(url, path)
    s = neurons.fetch(str(tmp_path), fetch_file)
    assert s["skipped"] == {str(order[0]): "not in the bucket"} and s["train"][0] == order[1]
