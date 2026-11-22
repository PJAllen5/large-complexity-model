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


def test_settings_can_be_overridden_for_longer_joint_training():
    env = "STAGEA_STEPS=160000 STAGEA_DATA=/shared/data"
    out = _bash(f'{env} bash -c \'source orca/stageA_settings.sh; echo "${{STAGEA_COMMON[*]}}"\'').stdout
    assert "--steps 160000" in out and "--data-dir /shared/data" in out
    default = _bash('echo "${STAGEA_COMMON[*]}"').stdout
    assert "--steps 20000" in default


def _fake_submit(tmp_path, mode, **env):
    """Run stageA_submit.sh with a stand-in sbatch that records its arguments."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "sbatch").write_text('#!/bin/bash\necho "$@" >> "$SBATCH_LOG"\necho 1000\n')
    (bin_dir / "sbatch").chmod(0o755)
    e = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", HOME=str(tmp_path),
             SBATCH_LOG=str(tmp_path / "sbatch.log"), **env)
    r = subprocess.run(["bash", "orca/stageA_submit.sh", mode], cwd=ROOT, env=e, capture_output=True, text=True)
    log = (tmp_path / "sbatch.log").read_text() if (tmp_path / "sbatch.log").exists() else ""
    return r, log


def test_specialists_long_submits_48_specialist_tasks(tmp_path):
    r, log = _fake_submit(tmp_path, "--specialists-long")
    assert r.returncode == 0, r.stderr
    root = tmp_path / "lcm-runs" / "stageA_spec160k"
    a30 = (root / "tasks_specialists-long_a30.txt").read_text().splitlines()
    l40s = (root / "tasks_specialists-long_l40s.txt").read_text().splitlines()
    assert len(a30) == 24 and all(t.startswith("8.5M ") for t in a30)
    assert len(l40s) == 24 and all(t.startswith("1.08M ") for t in l40s)
    assert all(t.split()[2].startswith("specialist_") for t in a30 + l40s)
    assert "160000 steps" in r.stdout and "--time=03:00:00" in log
    assert "stageA_reports" not in log                      # no Stage A report job in this mode


def test_specialists_long_skips_finished_runs(tmp_path):
    done = tmp_path / "lcm-runs" / "stageA_spec160k" / "8.5M_seed0" / "runs"
    done.mkdir(parents=True)
    (done / "specialist_boids.json").write_text("{}")
    r, _ = _fake_submit(tmp_path, "--specialists-long", TASK_TIME="02:00:00")
    assert "23 on A30, 24 on L40S" in r.stdout


def test_task_time_per_mode_and_override(tmp_path):
    (tmp_path / "a").mkdir()
    _, log = _fake_submit(tmp_path / "a", "--joint-long")
    assert "--time=03:30:00" in log
    (tmp_path / "b").mkdir()
    _, log = _fake_submit(tmp_path / "b", "--specialists-long", TASK_TIME="02:15:00")
    assert "--time=02:15:00" in log and "--time=03:00:00" not in log


def test_28m_specialists_on_a30_only(tmp_path):
    env = dict(STAGEA_A30_SIZES="28M", STAGEA_L40S_SIZES="", A30_MAX="8", TASK_TIME="03:30:00")
    r, log = _fake_submit(tmp_path, "--specialists-long", **env)
    assert r.returncode == 0, r.stderr
    root = tmp_path / "lcm-runs" / "stageA_spec160k"
    a30 = (root / "tasks_specialists-long_a30.txt").read_text().splitlines()
    assert len(a30) == 24 and all(t.startswith("28M ") for t in a30)
    assert "24 on A30, 0 on L40S" in r.stdout
    assert "--array=1-24%8 --gres=gpu:a30:1 --partition=short --time=03:30:00" in log
    assert "gpu:l40s" not in log                            # no L40S array submitted


def test_unknown_size_override_is_refused(tmp_path):
    r, log = _fake_submit(tmp_path, "--specialists-long", STAGEA_A30_SIZES="28m")
    assert r.returncode != 0 and "unknown size 28m" in r.stderr and log == ""


def test_joint_640k_uses_normal_partition(tmp_path):
    r, log = _fake_submit(tmp_path, "--joint-640k")
    assert r.returncode == 0, r.stderr
    tasks = (tmp_path / "lcm-runs" / "stageA_joint640k" / "tasks_joint-640k_a30.txt").read_text().splitlines()
    assert tasks == ["8.5M 0 joint", "8.5M 1 joint", "8.5M 2 joint"]
    assert "640000 steps, partition normal" in r.stdout
    assert "--partition=normal --time=07:00:00" in log and "gpu:l40s" not in log


def test_joint_weighted_passes_weights_to_every_job(tmp_path):
    r, log = _fake_submit(tmp_path, "--joint-weighted")
    assert r.returncode == 0, r.stderr
    assert "160000 steps, partition short, weights nbody=2.5,epidemic=2.5" in r.stdout
    assert "--partition=short --time=02:00:00" in log
    # the task and prep jobs source the settings with the exported STAGEA_WEIGHTS
    out = _bash('STAGEA_WEIGHTS=nbody=2.5,epidemic=2.5 bash -c \'source orca/stageA_settings.sh; '
                'echo "${STAGEA_COMMON[*]}"\'').stdout
    assert "--system-weights nbody=2.5,epidemic=2.5" in out
    assert "--system-weights" not in _bash('echo "${STAGEA_COMMON[*]}"').stdout
