"""Checks for orca/scaling_report.py on synthetic Stage A results with known answers."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orca"))
import scaling_report as sr  # noqa: E402


def _write_seed(root, size, seed, gap, overfit=False):
    """One <size>_seed<N> folder where joint CRPS = gap x specialist CRPS on every system."""
    d = os.path.join(root, f"{size}_seed{seed}", "runs")
    os.makedirs(d, exist_ok=True)
    m = lambda c, p=1.0: {"crps": c, "persist_crps": p, "prob_error": 0.2}  # noqa: E731
    val = [[1000, 0.0, -2.0], [2000, 0.0, -2.5], [3000, 0.0, -1.0 if overfit else -2.6]]
    runs = {"joint": {s: m(0.1 * gap) for s in sr.SYSTEMS}}
    runs.update({f"specialist_{s}": {s: m(0.1)} for s in sr.SYSTEMS})
    for s in sr.HELD_OUT:
        runs[f"loo_{s}"] = {s: m(0.5, 0.6 if s == "neurons" else 0.4)}   # beats baseline only on neurons
        runs[f"finetune_{s}"] = {s: m(0.2)}
        runs[f"scratch_{s}"] = {s: m(0.3)}                                 # 1.5x advantage
    for tag, ev in runs.items():
        json.dump({"tag": tag, "eval": ev, "loss": val if tag == "joint" else []},
                  open(os.path.join(d, f"{tag}.json"), "w"))


def _root(tmp_path, gaps, **kw):
    for size, per_seed in gaps.items():
        for seed, g in enumerate(per_seed):
            _write_seed(str(tmp_path), size, seed, g, **kw)
    return str(tmp_path)


def _verdict(s, label_start):
    return next(v for v in sr.verdicts(s) if v[0].startswith(label_start))


def test_gap_is_geometric_mean_of_joint_over_specialist(tmp_path):
    s = sr.summarize(sr.load_root(_root(tmp_path, {"8.5M": [1.2]})))
    assert s["8.5M"]["gaps"][0] == pytest.approx(1.2)
    assert s["8.5M"]["per_system"]["boids"] == pytest.approx(1.2)


def test_success_case(tmp_path):
    s = sr.summarize(sr.load_root(_root(tmp_path, {"1.08M": [1.29, 1.31], "8.5M": [1.14, 1.16],
                                                   "28M": [1.03, 1.05]})))
    assert _verdict(s, "Gap shrinks")[1] == "YES"
    assert _verdict(s, "Gap closes")[1] == "YES" and "28M" in _verdict(s, "Gap closes")[2]
    assert _verdict(s, "Stop")[1] == "NO"


def test_stop_case(tmp_path):
    s = sr.summarize(sr.load_root(_root(tmp_path, {"1.08M": [1.30, 1.31], "8.5M": [1.27, 1.28],
                                                   "28M": [1.25, 1.26]})))
    assert _verdict(s, "Gap shrinks")[1] == "NO"
    assert _verdict(s, "Gap closes")[1] == "NO"
    assert _verdict(s, "Stop")[1] == "YES - STOP"


def test_partial_results_say_not_yet(tmp_path):
    s = sr.summarize(sr.load_root(_root(tmp_path, {"1.08M": [1.3]})))
    assert _verdict(s, "Gap shrinks")[1] == "NOT YET" and _verdict(s, "Stop")[1] == "NOT YET"


def test_missing_run_leaves_seed_out_of_the_gap(tmp_path):
    root = _root(tmp_path, {"8.5M": [1.2, 1.4]})
    os.remove(os.path.join(root, "8.5M_seed1", "runs", "specialist_boids.json"))
    s = sr.summarize(sr.load_root(root))
    assert list(s["8.5M"]["gaps"]) == [0]


def test_transfer_and_overfitting_counts(tmp_path):
    s = sr.summarize(sr.load_root(_root(tmp_path, {"28M": [1.1, 1.1]}, overfit=True)))
    assert s["28M"]["zero_shot"]["neurons"] == (2, 2) and s["28M"]["zero_shot"]["epidemic"] == (0, 2)
    assert s["28M"]["few_shot"]["kuramoto"] == pytest.approx(1.5)
    assert s["28M"]["overfit"] == (2, 2)


def test_report_writes_markdown_and_chart(tmp_path):
    root = _root(tmp_path, {"1.08M": [1.3, 1.32], "8.5M": [1.2, 1.18]})
    sr.report(root)
    text = open(os.path.join(root, "SCALING.md")).read()
    assert "Gap shrinks" in text and "| 1.08M | 2 of 2 |" in text
    assert os.path.getsize(os.path.join(root, "scaling.png")) > 1000
