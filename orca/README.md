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

## 8. Summarise noise across all seeds (seconds, on the login node)

Once the seed jobs have finished (`squeue -u $USER` is empty), get the latest
code and summarise seed 0 (`results_orca/`) plus every seed in `~/lcm-runs/`.
This only reads small result files, so it's fine on the login node.

```bash
git pull
```

```bash
.venv/bin/python orca/noise_summary.py results_orca ~/lcm-runs/phase0_seed*
```

It prints each run's average and spread across seeds, the typical spread
per run type, how many seeds each headline finding holds in, pretraining's
few-shot advantage per seed, and a rough number of seeds the scale-up needs.
**Paste the whole output to Claude.**

---

## 9. Scale-up pilot (~20–40 min per GPU type)

Measures speed, memory and the effect of faster maths for each planned model
size, so the scale-up can be sized before ~100 runs are committed. Get the
latest code, then submit once per GPU type (both can run at the same time):

```bash
git pull
```

```bash
sbatch orca/pilot.sbatch
```

```bash
sbatch --gres=gpu:l40s:1 orca/pilot.sbatch
```

Each log (`orca/logs/lcm-pilot-<jobid>.out`) has a speed table and an
accuracy check. **Paste both logs to Claude.**

---

## 10. Stage A: overfitting check, then the full scale-up

All Stage A settings (bf16, 20k steps, 3 seeds, the three model sizes) live in
`orca/stageA_settings.sh`. Results go to `~/lcm-runs/stageA_n20k/<size>_seed<N>/`.
The submit script only submits jobs, so it's fine on the login node. It skips
tasks that already have results, so after a failure just run it again.

**10a. Overfitting check (~30 min):** seed-0 specialists for epidemic and
Lorenz96, at 8.5M on an A30 and 28M on an L40S.

```bash
git pull
```

```bash
bash orca/stageA_submit.sh --check
```

When `squeue -u $USER` is empty, read the validation curves and **paste the
output to Claude**:

```bash
.venv/bin/python orca/val_curves.py ~/lcm-runs/stageA_n20k/8.5M_seed0 ~/lcm-runs/stageA_n20k/28M_seed0
```

**10b. Full Stage A (only after Claude has reviewed 10a):** 108 tasks over
3 sizes and 3 seeds. At most 8 GPUs run at once, so this takes several
hours. The check's 4 runs are reused.

```bash
bash orca/stageA_submit.sh
```

Each task's log is `orca/logs/lcm-stageA-<arrayid>_<n>.out`. A report job
runs automatically once every task has succeeded. If any task fails, the
report job never starts: paste the failed task's log to Claude, then rerun
`bash orca/stageA_submit.sh`.

**10c. Scaling report (seconds, on the login node):** once every task has
finished, get the latest code and run the report. It judges the results
against the success criteria fixed before running, and writes
`~/lcm-runs/stageA_n20k/SCALING.md` and `scaling.png`.

```bash
git pull
```

```bash
.venv/bin/python orca/scaling_report.py
```

**Paste the whole output to Claude.** It also works on partial results:
sizes or seeds that aren't finished show as incomplete, and verdicts that
need them say "NOT YET".

---

## 11. Longer joint training (8x steps)

Tests whether the joint model was simply undertrained: it retrains **only
the joint model**, for 160,000 steps (8 systems x 20,000), at all 3 sizes and
3 seeds (9 runs, ~1.5-2 h). It reuses the Stage A data, and compares
against the Stage A specialists. Results go to `~/lcm-runs/stageA_joint160k/`.

```bash
git pull
```

```bash
bash orca/stageA_submit.sh --joint-long
```

It should print `Unfinished tasks: 6 on A30, 3 on L40S (... 160000 steps)`.
When `squeue -u $USER` is empty, check for failures, then run the report:

```bash
sacct -j <A30 array id>,<L40S array id> --format=JobID,State,Elapsed -X | grep -v COMPLETED
```

```bash
.venv/bin/python orca/scaling_report.py ~/lcm-runs/stageA_joint160k --specialists-from ~/lcm-runs/stageA_n20k
```

**Paste the report to Claude.** Its "Gap with the reference run's joint"
column shows the 20,000-step gaps side by side.

## 12. Specialists at 160k steps (fair comparison for step 11)

Step 11's joint models beat the 20k-step specialists, but those specialists
were still improving. This trains **the 8 specialists for 160,000 steps** at
1.08M and 8.5M, 3 seeds each (48 runs, ~40 GPU-h, ~6 h at 5 A30 + 3 L40S).
8.5M runs on A30s (like its joint model); 1.08M on L40Ss. Same data as
Stage A. Results go to `~/lcm-runs/stageA_spec160k/`.

First, see how long the 8.5M joint runs took (same step count and size as
an 8.5M specialist):

```bash
sacct -j 272979 --format=JobID,State,Elapsed -X
```

The default time limit per task is 3 h. If the 8.5M tasks (array tasks
4-6) took well under 2 h, use a tighter limit so jobs start sooner, e.g.
`TASK_TIME=02:00:00 bash orca/stageA_submit.sh --specialists-long`.
Otherwise:

```bash
git pull
```

```bash
bash orca/stageA_submit.sh --specialists-long
```

It should print `Unfinished tasks: 24 on A30, 24 on L40S (... 160000 steps)`.
When `squeue -u $USER` is empty, check for failures, then run the report:

```bash
sacct -j <A30 array id>,<L40S array id> --format=JobID,State,Elapsed -X | grep -v COMPLETED
```

```bash
.venv/bin/python orca/convergence_report.py
```

**Paste the report to Claude.** It gives the verdict per size (fixed before
running), the per-system gaps, how much the specialists gained from 8x
training, and whether any of them overfit.

## 13. 28M specialists at 160k steps (on A30s)

Does the gap keep falling with size (1.29 at 1.08M, 1.08 at 8.5M)? This
trains the 8 specialists at **28M** for 160k steps, 3 seeds (24 runs), and
compares them with the existing 28M joint models. The L40S nodes were down
on 2026-11-21, so this runs on **A30s**, 8 at a time (still within the
8-GPU cap). Each run should take ~2.6-2.7 h (pilot: 0.058 s/step), so the
time limit is 3.5 h (`short` allows 4 h). Expect ~64 GPU-h and ~8 h.

First check the L40S nodes. If they're back (`idle` or `mix`), tell Claude:
28M is ~2x faster there.

```bash
sinfo -N -n orcaga[01-06] -o "%N %T"
```

Otherwise:

```bash
git pull
```

```bash
STAGEA_A30_SIZES=28M STAGEA_L40S_SIZES= A30_MAX=8 TASK_TIME=03:30:00 bash orca/stageA_submit.sh --specialists-long
```

It should print `Unfinished tasks: 24 on A30, 0 on L40S (... 160000 steps)`
and submit one A30 array. After the first round (~3 h), check the run time
of the first tasks:

```bash
sacct -j <A30 array id> --format=JobID,State,Elapsed -X | head -12
```

When `squeue -u $USER` is empty, check for failures and run the same
report as step 12 (it now includes 28M):

```bash
sacct -j <A30 array id> --format=JobID,State,Elapsed -X | grep -v COMPLETED
```

```bash
.venv/bin/python orca/convergence_report.py
```

## 14. Why do n-body and epidemic lag? Two joint-model tests at 8.5M

At 28M the joint model matches the specialists on 6 of 8 systems; n-body
and epidemic make up the gap. The hypothesis is that they get too little
training inside the joint model. Two arms, both 8.5M, 3 seeds, on A30s:

- **A: 640k steps** (4x, ~80k per system). ~5.1 h per run, so it runs on the
  `normal` partition (1-day limit) with a 7 h task limit.
- **B: 160k steps, n-body and epidemic sampled 2.5x as often.** ~1.3 h per run.

Both now record validation NLL per system during training. They can run at
the same time (6 A30s):

```bash
git pull
```

```bash
bash orca/stageA_submit.sh --joint-640k
```

```bash
bash orca/stageA_submit.sh --joint-weighted
```

They should print `Unfinished tasks: 3 on A30, 0 on L40S (... 640000 steps,
partition normal)` and `(... 160000 steps, partition short, weights
nbody=2.5,epidemic=2.5)`. When `squeue -u $USER` is empty, check for
failures, then run one report per arm:

```bash
sacct -j <640k array id>,<weighted array id> --format=JobID,State,Elapsed -X | grep -v COMPLETED
```

```bash
.venv/bin/python orca/convergence_report.py --joint ~/lcm-runs/stageA_joint640k --compare-joint ~/lcm-runs/stageA_joint160k --out ~/lcm-runs/stageA_joint640k
```

```bash
.venv/bin/python orca/convergence_report.py --joint ~/lcm-runs/stageA_jointw160k --compare-joint ~/lcm-runs/stageA_joint160k --out ~/lcm-runs/stageA_jointw160k
```

**Paste both reports to Claude.** Each has the pre-registered hypothesis
check, per-system gaps next to the 160k joint model, and the step where
each system's validation NLL was best.

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
