# Running LCM on ORCA

Step-by-step guide for rerunning Phase 0 on Portland State's ORCA cluster.
Run each command yourself, and paste the output Claude asks for back into the
chat.

**Goal of the first ORCA run:** reproduce the laptop's Phase 0 results. If
they match, the setup is right, and later changes in results come from the
model, not the machine. **Pass** = every forecast score (CRPS) within 10% of
the laptop's, and the same three headline answers. `orca/compare.py` checks
this automatically.

**Rule:** the login node is only for light work: editing, git, installing
packages, submitting jobs. Anything that computes goes through `sbatch`.

---

## 1. Log in and get the code

```bash
ssh pjallen5@login.orca.pdx.edu
```

```bash
git clone https://github.com/PJAllen5/large-complexity-model.git && cd large-complexity-model
```

## 2. Check your limits (paste the output to Claude)

```bash
sinfo
```

```bash
sacctmgr show assoc user=$USER format=user,account,partition,grptres,maxtres,maxjobs
```

```bash
module avail cuda python
```

## 3. One-time setup (~5–10 min, on the login node)

```bash
bash orca/setup.sh
```

The last line should say `Setup done`, with a line before it showing the
torch version and `built for CUDA 12.x` (or similar).

## 4. GPU check job (~5 min)

```bash
sbatch orca/check_gpu.sbatch
```

It replies `Submitted batch job <number>`. **Note the number**: it's the job ID,
and the log file is named after it. The job may wait in the queue before
starting. Watch it with the first command below. `PD` means waiting and `R`
means running; when it's gone from the list, it's finished. Then read its log
with the second command, using your job ID.

```bash
squeue -u $USER
```

```bash
cat orca/logs/lcm-check-<jobid>.out
```

You should see `PASS`, a "training step" speed, and the tests passing (e.g.
`18 passed`). **Paste the whole log to Claude.**

## 5. Phase 0 rerun (~17 min on an A30, measured 2026-11-19)

Only once step 4 passes:

```bash
sbatch orca/phase0.sbatch
```

Note the job ID again. Once `squeue` shows it as `R` (the log file only
appears when the job starts), follow along with the command below. Ctrl-C
stops watching, not the job.

```bash
tail -f orca/logs/lcm-phase0-<jobid>.out
```

The log ends with a comparison table and one of three verdicts:
`PASS`, `PASS WITH WARNINGS` (only noisy runs or near-ties moved) or `FAIL`.
**Paste the end of the log to Claude**, from `== Experiment took` onward.

To use the faster L40S instead of an A30:

```bash
sbatch --gres=gpu:l40s:1 orca/phase0.sbatch
```

## 6. Bring the results back to your laptop

Run this **on your laptop**, not on ORCA. It copies the report, charts and
metrics, and leaves the bulky data and checkpoints behind:

```bash
rsync -av --exclude data --exclude checkpoints pjallen5@login.orca.pdx.edu:large-complexity-model/results_orca ~/large-complexity-model/
```

Then ask Claude to log and commit them.

---

## 7. Seed-noise run (measures run-to-run chance, ~17 min)

Reruns Phase 0 with a different training seed (same data, same GPU type)
and compares it with the first ORCA run. This is a **measurement, not a
pass/fail test**: it shows how much scores move by chance, so later
"improvements" can be judged against that.

First make sure ORCA has the latest code:

```bash
cd ~/large-complexity-model && git pull
```

```bash
sbatch orca/seed_noise.sbatch
```

The log is `orca/logs/lcm-seed-<jobid>.out`. It ends with a noise table:
typical and largest score changes for the main runs and the few-shot runs,
and any headline answers that flipped. **Paste the end of the log to
Claude**, from `== Experiment took` onward.

Results go to `~/lcm-runs/phase0_seed1/` on ORCA, **outside the repo**, so
they never block a later `git pull`. To copy them to your laptop (run this
on the laptop), use:

```bash
rsync -av --exclude data --exclude checkpoints pjallen5@login.orca.pdx.edu:lcm-runs/phase0_seed1/ ~/large-complexity-model/results_orca_seed1/
```

For another seed: `sbatch --export=ALL,SEED=2 orca/seed_noise.sbatch`.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| `Could not load module python/...` | Run `module avail python`, then e.g. `export LCM_PYTHON_MODULE=python/3.12.<x>` and rerun the step |
| GPU check says `CUDA available: False` | The installed PyTorch is too new for ORCA's NVIDIA driver. Paste the log to Claude, which will give the exact reinstall command for the driver version shown by `nvidia-smi` |
| Job sits in `PD` for a long time | `squeue -u $USER` shows the reason. `Resources` means the GPUs are busy, so wait or switch GPU type. A limit-related reason means you've hit your account's limits |
| Job hit its time limit | Submit `orca/phase0.sbatch` again. Finished runs are cached and skipped |
| Log says `Submit this job from the repo root` | `cd ~/large-complexity-model`, then submit again |
| Log says `run 'bash orca/setup.sh' first` | Do step 3, then submit again |
| `git pull` says untracked files "would be overwritten" | ORCA has its own copy of results that were since committed (this happened with `results_orca/`). Move ORCA's copy aside (`mkdir -p ~/backup && mv <the listed files> ~/backup/`), `git pull`, then check they're identical with `diff -r` |
| Job finished but there's **no log file** | Slurm couldn't write the log, usually because the job was submitted from inside `orca/` or another folder. Run `sacct -j <jobid>` to see its state, then `cd ~/large-complexity-model` and submit again |
