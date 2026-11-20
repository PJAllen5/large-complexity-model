"""Checks for the Stage A helpers: overfitting detection and the shell task bookkeeping."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orca"))
import val_curves  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _pts(vals):
    return [(1000 * (i + 1), 0.0, v) for i, v in enumerate(vals)]


def test_rising_validation_is_flagged():
    step, best, up, verdict = val_curves.assess(_pts([1.0, 0.8, 0.7, 0.75, 0.85]))
    assert step == 3000 and verdict.startswith("OVERFITTING")


def test_still_improving_is_reported():
    assert "still improving" in val_curves.assess(_pts([1.0, 0.9, 0.8]))[3]


def test_small_wobble_is_not_overfitting():
    assert val_curves.assess(_pts([1.0, 0.8, 0.79, 0.81]))[3].startswith("flat")


def test_runs_without_validation_are_skipped():
    assert val_curves.curve({"loss": [[100, 1.2], [200, 1.1]]}) == []


def _bash(cmd):
    return subprocess.run(["bash", "-c", f"source orca/stageA_settings.sh; {cmd}"], cwd=ROOT,
                          capture_output=True, text=True)


def test_size_names_map_to_model_shapes():
    assert _bash("stageA_model_args 8.5M").stdout.strip() == "--d 256 --layers 8 --heads 8"
    assert _bash("stageA_model_args 28M").stdout.strip() == "--d 384 --layers 12 --heads 8"
    assert _bash("stageA_model_args 99M").returncode != 0


def test_transfer_task_produces_three_runs():
    out = _bash("stageA_task_outputs transfer_epidemic").stdout.split()
    assert out == ["loo_epidemic", "finetune_epidemic", "scratch_epidemic"]
    assert _bash("stageA_task_outputs joint").stdout.split() == ["joint"]
