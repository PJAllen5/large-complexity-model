# LCM project log

A running record of every decision, test and result on the Large Complexity
Model. Newest entries go at the bottom of the timeline. Each entry says
**what** was decided or run, **why**, and **what came out of it**.

- Code: [README.md](README.md)
- Latest results: [results/REPORT.md](results/REPORT.md)

---

## Reference: model size and settings

### What "parameters" means

A model's **parameters** (or *weights*) are the numbers it adjusts during
training. Think of them as millions of tiny dials, each nudged a little every
time the model's forecast turns out wrong. Together they hold everything the
model has learned about how systems behave. More parameters means more
capacity to learn many different systems, but also more compute and data
needed to train.

**Settings** (*hyperparameters*) are the choices *we* make before training,
such as model width, number of layers, how long to train and how much data
to use. The model never changes them.

### Where the parameters live (Phase 0 model, 1.08M total)

| Part | Parameters | What it does |
|---|---|---|
| Input embedding | ~6,000 | Turns each chunk of 4 time steps of one entity into a vector of 128 numbers |
| Transformer blocks (×4) | ~1,058,000 (98%) | The "thinking" part. Each block looks back along every entity's history, then across all entities at the same moment, then processes what it found |
| Output head | ~17,000 | Turns the result into a forecast: a mix of 4 bell curves over the next 4 time steps |
| Interaction bonuses | 32 | Learned nudges telling attention to favour an entity itself and its direct neighbours |

### Settings used in Phase 0 vs planned for larger runs

The larger sizes are **proposals, not yet decided**. Parameter counts were
measured by building each model in code.

| Setting | Phase 0 (run) | Scale-up (planned) | Phase 1 start (planned) |
|---|---|---|---|
| **Parameters** | **1.08M** | **~8.5M** | **~28–50M** |
| Width (numbers per token) | 128 | 256 | 384–512 |
| Layers (transformer blocks) | 4 | 8 | 12 |
| Attention heads | 4 | 8 | 8 |
| Mixture components (bell curves per forecast) | 4 | 4–8 | 8 |
| Systems trained on | 8 simulated | ~20 simulated | simulated + 2–3 real datasets |
| Episodes per system | 2,000 | ~5,000 | varies with the dataset |
| Max entities / features per system | 12 / 4 | 12 / 4 | needs lifting to hundreds of entities |
| Time steps: history shown / forecast | 48 / 16 | 48 / 16 | to be decided per dataset |
| Patch (time steps per token) | 4 | 4 | 4–8 |
| Training steps per model | 5,000 | ~30,000 | ~100,000+ |
| Batch size (episodes per step) | 32 | ~128 | ~256 |
| Learning rate | 3e-4 (1e-4 for fine-tuning) | 3e-4 | tuned |
| Hardware | MacBook Air M3, 8-core GPU | 1× RTX 4090 | 1× A100/H100 80 GB |
| Time / cost | 82 min / free | ~10–30 h / ~$20–100 | 100s–1000s GPU-h / ~$500–5,000 |

---

## Timeline

### 2026-10-24: Concept and architecture

**Goal.** Build a "Large Complexity Model" (LCM): like an LLM, but trained on
complex-systems data, and predicting the likelihood of outcomes. Long-term
scope runs from quantum to cosmological scales, both for single systems and
for systems linked together.

**Decisions**
- **Universal data format:** every system is *entities × time × features*,
  plus a graph of who interacts with whom. *Why:* LLMs succeeded because all
  text shares one format. This is the equivalent for complex systems.
- **Probabilistic output:** the model predicts a distribution, not a single
  value, and sampled futures give the odds of outcomes. *Why:* the core goal
  is likelihoods.
- **Stacked by scale, not a single quantum-to-cosmos model:** each level
  learns its own dynamics and passes summaries to the levels above and below.
  *Why:* each scale in nature has its own effective rules, and no model can
  simulate a whale from quarks.
- **Phased roadmap:**
  - Phase 0: simulators on a laptop
  - Phase 1: real data
  - Phase 2: linking scales
  - Phase 3: scale up compute
- **Start with simulators.** *Why:* the data is free and unlimited, and the
  true answer is known, so the core idea can be tested cheaply.

### 2026-10-24: Phase 0 build

**Built**
- 8 stochastic simulators, each with fresh random parameters per episode:
  - predator–prey food web (ecology)
  - boids flocking (animal behaviour)
  - Kuramoto oscillators (synchronisation)
  - Lorenz96 (climate/chaos)
  - n-body gravity (astronomy)
  - gene regulatory network (cell biology)
  - FitzHugh–Nagumo neurons (neuroscience)
  - SIR epidemic across cities (society)
- **Model:** a transformer reading tokens of one entity × 4 time steps,
  outputting a mix of 4 bell curves over the next chunk. It is **never told
  which system it is looking at**; it has to infer that from context.
- **Evaluation metrics:**
  - CRPS: accuracy of the whole predicted distribution
  - 90% interval coverage
  - comparison against a "nothing changes" baseline
  - an **outcome-odds test**, where the simulator replays each situation 64
    times to get the true odds of "will this quantity rise?"
- **Experiment design:**
  1. One joint model vs 8 specialists at the same training budget.
  2. Zero-shot: hide neurons, epidemic or Kuramoto during training, then
     forecast it.
  3. Few-shot: 64 examples of the hidden system, pretrained model vs a fresh one.

**Issues found and fixed during build**
- **Lorenz96 blew up numerically.** Simple Euler steps were unstable at high
  forcing, so it was switched to RK4 integration.
- **Crash when the random warm-up was 0 steps.** The simulator's run function
  now handles zero steps.
- **Training too slow** (0.21 s/step):
  - Switched full attention over all 192 tokens to **axial attention** (time
    within each entity, then across entities at the same moment): 0.16 → 0.13 s/step.
  - Profiling showed the model is compute-bound; half precision gave no gain on the M3.
  - **Batch 64 → 32** halved step time to 0.065 s.
- **Epidemics broke normalisation.** A flat context (no infections yet)
  followed by an outbreak produced huge scaled values that were clipped at ±10.
  Replaced the clip with an **asinh squash** (linear near zero, logarithmic for
  big jumps), so no information is lost.

**Quick test run** (`--quick`: 300 training episodes, 300 steps, ~7 min)
- The pipeline works end to end.
- The models were undertrained, but transfer was already visible: pretrained
  plus a few examples beat a fresh model on 2 of 3 held-out systems.

### 2026-10-24: Project moved and put under version control

- Moved from a temporary folder to `~/large-complexity-model`.
- **Private** GitHub repo: <https://github.com/PJAllen5/large-complexity-model>.
- `.gitignore` excludes regenerable files: simulation data, model
  checkpoints, the Python environment and logs. Reports, charts and per-run
  metrics are committed.
- The first full run was stopped partway through, when the joint model was
  ~80% trained, so it wouldn't write into the old folder. It was then
  restarted from scratch in the new folder.

### 2026-10-24: Phase 0 full run (results)

**Setup:** 2,000 episodes per system, 5,000 steps per model, 128 test episodes
per system. 20 models trained in total. 82 minutes on the MacBook Air M3. The
Mac was kept awake with `caffeinate`.

| Question | Result |
|---|---|
| Does the joint model beat specialists? | **No: 0 of 8.** Close on epidemics (0.041 vs 0.040) and ecology. Below the baseline on gene networks |
| Zero-shot on an unseen system? | **Weak: 1 of 3** (neurons only) |
| Does pretraining help learn new systems from 64 examples? | **Yes: 3 of 3.** Epidemic error 0.10 vs 0.25 for a fresh model |
| Correct odds? | **Yes, mostly.** Average odds error 0.24 vs 0.40 for an informed guess. Better on 7 of 8 systems (all except gene networks) |
| Calibration | Truth inside the 90% range **87%** of the time (ideal 90%) |

**Interpretation:** this is the pattern expected from a model that is too
small. 1.08M parameters are shared across 8 systems, while each specialist
spends them all on one.

**Next test:** scale up the model and the number of systems, and check
whether the gap to the specialists closes.

### 2026-10-25: Compute planning

**Decisions**
- **Defer the scale-up and Phase 1 to rented cloud GPUs**, starting with
  money available Friday (2026-10-30).
- **Don't run Phase 1 on the laptop.** *Why:* it needs hundreds to thousands
  of GPU-hours (weeks to months on an M3) and more disk than the laptop has
  free (34 GB).
- **The user declined to build Phase 1 data converters on the laptop.** The
  whole pipeline, from converters to training, will run on a large cloud
  machine instead.

**Cost estimates** (prices October 2026, not yet tested):

| Stage | Hardware | GPU time | Cost |
|---|---|---|---|
| Scale-up | 1× RTX 4090, ~100 GB disk | 10–30 h | $20–100 |
| Phase 1 | 1× A100/H100 80 GB, 200–500 GB disk | 100s–1000s h | $500–5,000 |

**Plan for the cloud run**
- Rent one GPU machine.
- Clone the repo and run one pipeline command.
- Push the results back to GitHub.
- Shut the machine down.

Option: do downloading and converting on a cheap CPU-only machine, writing to
a shared volume, then attach a GPU machine only for training.

**Laptop reference:** MacBook Air (Mac15,12), Apple M3 with an 8-core GPU,
16 GB shared memory, ~2.8 TFLOPS FP32, 100 GB/s memory bandwidth, fanless.

### 2026-10-28: Parameter count corrected, project log started

- **Correction:** the trained Phase 0 model has **1.08M** parameters, not the
  0.8M quoted earlier. The 0.8M figure came from before the switch to axial
  attention, which adds a second attention layer to every block. The README
  has been corrected.
- Measured parameter counts for candidate larger sizes:

  | Width | Layers | Parameters |
  |---|---|---|
  | 256 | 8 | 8.47M |
  | 384 | 12 | 28.47M |
  | 512 | 12 | 50.54M |

- **Decision:** document every decision, test and result in this file from
  now on. `CLAUDE.md` tells future sessions to keep it updated.

### 2026-10-29: Tests and CI added

- **What:** first test suite (`tests/test_lcm.py`, 18 tests, ~2 s on CPU) and a
  GitHub Actions workflow that runs it on every pull request and push to main.
- **What the tests cover:** every simulator's output shape, masks and symmetric
  interaction graph; reproducible generation; normalisation using only the
  context window (no future leakage); the mixed sampler drawing every domain;
  the 1,081,380 parameter count; forward/loss/gradient sanity; causality in
  time; empty entity slots having no effect on real entities; `sample_future`
  shape; and a tiny model's loss falling over 60 training steps.
- **Why:** nothing checked the code before. Leakage or a broken mask would make
  experiment results meaningless without any visible error.
- **Check that the tests bite:** two bugs were planted by hand and both were
  caught: turning off causal attention failed the causality test, and
  normalising over the whole episode failed the leakage test.
- **Not covered:** forecast quality and `evaluate.py` metrics. Those come from
  `experiment.py` runs, which are too slow for CI.

### 2026-11-19: ORCA access and specs

- **What:** PJ has access to **ORCA** (Oregon Regional Computing Accelerator),
  an NSF-funded GPU cluster hosted at Portland State. **Access is free.** Specs
  below are from the public docs at <https://orca.pdx.edu>, read on this date.
  Nothing has been run on ORCA yet.

| Node type | Nodes | GPUs per node | GPU | GPU memory | CPU | RAM |
|---|---|---|---|---|---|---|
| orca01–06 | 6 | 4 | NVIDIA L40S (Ada) | 48 GB each, 192 GB per node | 64-core AMD EPYC 9534 | 576 GB |
| orca07–25 | 19 | 4 | NVIDIA A30 (Ampere) | 24 GB each, 96 GB per node | 64-core AMD EPYC 9534 | 576 GB |

- **GPU speed (FP32):**
  - L40S: 91.6 TFLOPS, 864 GB/s
  - A30: 10.3 TFLOPS, 933 GB/s
  - For comparison, the M3 laptop is ~2.8 TFLOPS.
  - The L40S has no hardware FP64, which doesn't matter here because LCM
    trains in FP32.
- **Scheduler:** Slurm.
  - Partitions: `short` (4 h limit), `normal` (7 days), `long` (20 days).
  - Jobs get **no GPU** unless they request one, for example
    `--gres=gpu:l40s:1`.
- **Storage:**
  - `/home`: 100 GB, backed up nightly.
  - `/scratch` workspaces: fast, not backed up, 30-day leases via
    `ws_allocate`, at most 5 extensions.
  - Per-node local SSD: 480 GB at `/tmp`, wiped when the job ends.
- **Software:**
  - Rocky Linux, Python modules (3.12, 3.14), `uv`, `venv`.
  - Conda is discouraged.
  - The CUDA page was not read.
- **Login:** `ssh <user>@login.orca.pdx.edu`, SSH keys only. Don't compute
  on the login node. Open OnDemand (web and Jupyter) is also available.
- **Not in the public docs:**
  - per-user GPU or job limits
  - interconnect
  - scratch quota and purge rules
  - container (Apptainer) support
  - CUDA versions

  Check these on the cluster with `sinfo`,
  `sacctmgr show assoc user=$USER` and `module avail cuda`.
- **What it means for LCM (not yet decided):**
  - One L40S (48 GB, ~30× the laptop's FP32 throughput) is enough for the
    scale-up (~8.5M parameters) and for early Phase 1 models (28–50M).
  - The 20 Phase 0-style runs are independent, so they could run in parallel
    as a Slurm job array.
  - Free access may replace the paid rental planned in the 2026-10-25 entry.

### 2026-11-19: ORCA run plan and scripts (Phase 0 rerun)

**Decisions**
- **Option B (PJ runs, Claude writes):** PJ runs every command on ORCA
  (username `pjallen5`) and pastes output back. Claude has no access to the
  cluster. *Why:* PJ sees every command, and no extra setup is needed. The
  alternative, an SSH connection from the Claude app, was declined for now.
- **Repo made public** by PJ, so ORCA can `git clone` without GitHub keys.
  Results come back by `rsync` to the laptop and are committed from there.
  Note: commit author emails are now publicly visible.
- **First ORCA job reproduces Phase 0 rather than starting the scale-up.**
  *Why:* if the same experiment gives the same answers on ORCA, the setup is
  trustworthy, and later changes can be put down to the model.
- **Pass criterion, fixed before running:** every CRPS within **10%** of the
  laptop run and the **same three headline answers** (0/8, 1/3, 3/3).
  `orca/compare.py` checks this. The simulated data is identical (same
  seeds), so only GPU arithmetic and random sampling differ.
- **First run on an A30** (`short` partition, 3 h limit, 1 GPU). There are
  more A30s, so jobs should start sooner. Switch to an L40S with
  `--gres=gpu:l40s:1`. The run stays sequential on one GPU; parallel job
  arrays are left for the scale-up.
- **Python 3.12 module** on ORCA (`python/3.12.12-gcc-13.4.0`) instead of
  3.14, as the safer bet for PyTorch's CUDA builds.

**Built** (`orca/`)
- `setup.sh`: one-time venv and package install on the login node.
- `env.sh`: loads ORCA's Python module and sets the import path.
- `check_gpu.sbatch` and `gpu_check.py`: GPU visible? correct maths? step
  speed compared with the M3? then the test suite.
- `phase0.sbatch`: full experiment into `results_orca/`, then the comparison.
- `compare.py`: per-run CRPS differences plus the headline answers, ending
  with PASS or FAIL.
- `README.md`: step-by-step guide.
- `.gitignore` now excludes `orca/logs/`.

**Tested on the laptop** (no Slurm, CUDA or `module` here, so `module` and
`nvidia-smi` were stand-ins):
- All four shell scripts pass a syntax check.
- `compare.py` gives PASS for the laptop run against itself; FAIL for the
  laptop run against the undertrained quick run (all three headline answers
  differ); and lists missing runs and FAILs on a partial folder.
- A dry run of `check_gpu.sbatch` **caught a bug**: the import-path line in
  `env.sh` failed under `set -u` (`BASH_SOURCE` unbound), so `lcm` couldn't
  be imported. Fixed by using the repo root, since every script `cd`s there
  first. After the fix the check reaches the GPU test and stops cleanly with
  "CUDA available: False", as expected on a Mac.
- A dry run of `phase0.sbatch`, with the experiment replaced by a stub,
  completes and prints PASS.
- The existing 18 tests still pass.

**Not tested:** anything on ORCA itself: the module name, the PyTorch/CUDA
driver match, Slurm directives, GPU speed and the real rerun. Those come from
PJ's first jobs.

### 2026-11-19: First ORCA login (username mix-up)

- **Problem:** `ssh pallen1@login.orca.pdx.edu` and `pallen1_sou@…` both
  failed with `Permission denied (publickey)`. A verbose ssh run showed the
  laptop *did* offer its key (`~/.ssh/id_ed25519`), so ORCA had no key on
  file for those usernames.
- **Cause:** `pallen1` is PJ's **ACCESS-CI** sign-in for the Registry website
  (`pallen1@access-ci.org`). The **cluster username is `pjallen5`**, listed
  under *Identifiers → Network* in the Orca Registry profile.
- **Result:** `ssh pjallen5@login.orca.pdx.edu` works, with 0 of 100 GB home
  used. `orca/README.md` and the plan entry above now say `pjallen5`.
- **Note:** ORCA's docs don't say where to find the username or how to fix
  this error. Support is via the ORCA Zulip chat.

### 2026-11-19: Code review of the ORCA scripts, and fixes

PR #3 (ORCA scripts) was merged before its code review finished. The review
found 8 issues, all fixed in this change. PJ asked to fix them before
continuing the post-merge questions. Two more bugs turned up while testing
the fixes.

**Fixed**
1. **The pass check couldn't tell noise from real failure.** Requiring all
   25 scores within 10% and identical headline answers was never checked
   against normal run-to-run randomness. Epidemic joint vs specialist
   (0.041 vs 0.040) could flip by chance and make a working setup FAIL.
   **Decision:** three verdicts.
   - `FAIL`: a main run (joint, specialist, never-seen-it) moved more than
     10%, a headline answer flipped on a clear gap, runs are missing, or
     the data differs.
   - `PASS WITH WARNINGS`: only the few-shot runs (64 training episodes)
     moved more than 10%, or a headline flipped on a near-tie (within 5%
     in either run).
   - `PASS`: otherwise.

   The 10% and 5% thresholds are **judgement calls, not measured noise**.
   Running the same experiment with a second seed would measure the real
   spread; not done yet.
2. **Simulated data wasn't confirmed identical.** `compare.py` now checks
   the generated data against `results/data_checksums.json` (sha256 of the
   laptop's array contents, committed). Regenerating all 16 datasets on the
   laptop reproduced the checksums exactly.
3. **ORCA's module system could crash the jobs under `set -euo pipefail`.**
   `env.sh` now relaxes `-e`/`-u` while loading modules and restores them
   afterwards. `module` errors are no longer hidden. Every script now stops
   with `source orca/env.sh || exit 1`.
4. **`orca/logs/` was missing on a fresh clone.** Slurm doesn't create it,
   so jobs would fail with no log. The folder is now kept in git
   (`.gitkeep`).
5. **Unclear failure when a job was submitted from the wrong folder or
   before setup.** Both job scripts now check first and print what to do.
   The README troubleshooting rows were corrected: a job with no log means
   check `sacct -j <jobid>`.
6. **CPU thread oversubscription.** Threads are now capped at the job's
   CPUs (`OMP_NUM_THREADS` etc.).
7. **The import path depended on the current folder.** `env.sh` now uses
   its own location, falls back to the current folder, and warns if that
   isn't the repo.
8. **The README's `lcm-*-*.out` log commands** could match no log or
   several. They now use the job ID that `sbatch` prints. The runtime
   estimate is labelled "not yet measured".

**Bugs found while testing the fixes** (both fixed)
- Saving shell settings with `$(set +o)` silently dropped `-e`. Bash resets
  it inside `$(...)`, so a failed `module load` printed a warning and the
  job **kept going**. Now saved from `$-` directly.
- On the Mac's bash 3.2, the repo-root lookup returned `/`. It now falls
  back to the current folder and warns.

**Tested on the laptop** (stand-in `module` and `nvidia-smi`):
- A strict script sourcing `env.sh` keeps `-e`/`-u` and stops on a later
  error.
- A failed module load stops the job and shows the module error.
- Both guards print their messages.
- `check_gpu.sbatch` reaches the GPU test.
- `phase0.sbatch` (experiment stubbed) reports data identical and PASS.
- `compare.py` scenarios: identical → PASS; epidemic near-tie flip →
  PASS WITH WARNINGS; few-shot +30% → PASS WITH WARNINGS; joint lorenz96
  +20% → FAIL; one changed number in the data → FAIL.
- 18/18 tests pass.

**Not tested:** anything on ORCA itself.

**Process note:** a 424-line PR was merged before its review finished. Next
time, wait for the review before merging.

### 2026-11-19: ORCA environment confirmed, GPU check passed

**Cluster facts from PJ's account** (these differ from the public docs in places):
- Node names are `orcaga01–25`, not `orca01–25` as in the docs.
- Time limits: `short` 4 h, `normal` **1 day**, `long` **7 days**. The
  docs say 7 and 20 days. Plan scale-up jobs to fit within 1 day, or use
  `long`.
- The `osg` partition is in use by Open Science Grid jobs. When checked,
  13 nodes were idle in `short`.
- `sacctmgr` shows **no per-user limits** on record for `pjallen5`.
- Modules: `python/3.12.12-gcc-13.4.0` and `python/3.14.0` (default);
  `cuda/11.8.0` and `cuda/12.9.0` (default, loaded).
- NVIDIA driver **610.57.04**, which supports CUDA up to 13.3.

**Setup** (`bash orca/setup.sh`): torch **2.15.0+cu132** (CUDA 13.2
build), numpy **2.5.5**. The laptop has torch 2.14.1 and numpy 2.5.4.
- The concern that the CUDA 13.2 build would need a newer driver than
  ORCA's CUDA 12.9 module didn't apply: the driver is 610.

**GPU check** (job 271910, A30 on `orcaga07`, ran 30 s): **PASS**.
- CUDA available.
- GPU vs CPU matmul difference: 5.05e-05.
- One Phase 0 training step: **0.0150 s**, against 0.065 s on the M3
  (**4.3× faster**).
- 18/18 tests passed.

**Next:** the Phase 0 rerun (`sbatch orca/phase0.sbatch`). At 4.3× the
step speed, it should take roughly 20–30 min, not 82 min. That's an
estimate; evaluation time hasn't been measured on the A30.

### 2026-11-19: Phase 0 reproduced on ORCA

**Run:** job 271911, `orca/phase0.sbatch`, one **A30** on `orcaga07`.
Same config as the laptop run (2,000 episodes per system, 5,000 steps,
20 models). **17.5 min** (laptop: 82 min, so 4.7× faster). Cost: free.
Results are in [results_orca/REPORT.md](results_orca/REPORT.md).

**First verdict: `FAIL`, caused by the data check, not ORCA.** All
25 CRPS scores were within 2% and the headline answers matched, but the
bit-for-bit data checksums differed for boids, Lorenz96 and n-body.

**Investigation:** PJ copied `results_orca/` to the laptop with `rsync`, and
the two data sets were compared directly.
- 10 of 16 files are bit-identical: epidemic, Kuramoto, gene network,
  neurons and predator–prey.
- In the 6 boids, Lorenz96 and n-body files:
  - at most 0.5% of values differ;
  - the max difference is 4.8e-4, which is 1.6e-4 of the data spread;
  - differences are zero at the first time step and grow over time;
  - masks and graphs are identical.
- **Cause:** last-digit floating-point rounding that differs between CPUs
  (Apple M3 vs AMD EPYC) and builds up over the physics simulators' many
  steps. Discrete and simpler simulators reproduce exactly.

**Fix:** `compare.py` now accepts data that is bit-identical, **or** whose
per-episode summary (mean, std, min, max) is within **0.1% of the data
spread**, with identical masks and graphs.
- Reference summaries are in `results/data_fingerprint.npz` (226 KB,
  float32), alongside `results/data_checksums.json`.
- New tests in `tests/test_compare.py` (now 25 tests in total):
  - **accepted:** identical data; rounding-sized noise;
  - **rejected:** a different seed; one episode simulated differently; a 1%
    systematic change; a changed graph.
- **Known blind spot**, written up as a test: a single changed value out of
  ~3,000 per episode isn't detected. The realistic causes (seed, simulator
  or library changes) alter many values at once.
- An earlier version of the new test asked for a single corrupted value to
  be caught. It wasn't, which is how the blind spot was found.

**Final verdict: `PASS`.**
- Data matches: 10 files bit-identical, 6 within rounding (worst 7.3e-6 of
  the spread).
- 25/25 CRPS within 10%; the largest difference was **1.8%** (`loo_neurons`).
- Headline answers are identical: joint beats specialist **0/8**,
  zero-shot beats baseline **1/3**, pretraining helps few-shot **3/3**.

**What this tells us**
- The ORCA setup is trustworthy, so later result changes can be put down to
  the model.
- Moving to a different machine, with the same seeds, moves scores by
  ≤ 1.8%. That's the noise from hardware alone. **It is not the
  seed-to-seed spread**, which is still unmeasured and probably larger,
  especially for the 64-episode few-shot runs. A second-seed run on ORCA
  (~17 min, free) would measure it before the scale-up.

### 2026-11-19: Second-seed noise run set up

**Why:** before the scale-up, we need to know how much results move by chance
alone. Otherwise a "better" bigger model could just be luck. Comparing ORCA
with the laptop only measured *hardware* noise (≤ 1.8%), not training noise.

**Decisions**
- **`experiment.py --seed N`** changes only the training randomness:
  starting weights and batch order. The simulated data (fixed seeds) and
  the true-odds replays (seed 999) stay fixed. The default is 0, so earlier
  runs are unchanged. Each run's JSON now records its seed.
- **Compare seed 1 with seed 0 on ORCA** (same hardware, same data), so
  the differences are pure training noise.
- **`orca/seed_noise.sbatch`** writes results to `~/lcm-runs/phase0_seed<N>/`,
  **outside the repo**. *Why:* committing `results_orca/` from the laptop
  made ORCA's later `git pull` refuse to overwrite its own untracked copy.
  It was fixed by moving the files aside and checking with `diff -r` that
  they were identical.
- **`compare.py --noise A B`** gives a measurement, not a verdict:
  - typical and largest |CRPS| and odds-error changes, separately for main
    runs and few-shot runs;
  - the largest movers;
  - headline answers, with any flips;
  - a data check against the laptop reference.
- Data-check message fix: a size mismatch now says "different size: N vs M
  episodes" instead of "masks/graphs differ".

**Tested on the laptop**
- `seed_noise.sbatch` ran end to end with stand-in `module` and
  `nvidia-smi`, using `--quick` with seed 1 (7 min). The job wrote outside
  the repo, recorded `"seed": 1`, and printed the noise report.
- 25/25 tests pass.
- In quick mode (300 training steps, badly undertrained), seed 1 vs seed 0
  moved main-run CRPS by **4.7% median and 41.5% max** (epidemic
  specialist). One headline answer flipped: joint beat the epidemic
  specialist under seed 0 only. These numbers come from undertrained
  models and **don't predict** the full-size noise. They do show that seed
  noise can be much larger than hardware noise, which is why this run
  matters.

**Not tested:** the full-size noise run on ORCA. That's PJ's next job
(`orca/README.md` step 7).

**Correction:** Phase 0 trains **18 models** (1 joint + 8 specialists + 3
never-seen-it + 3 fine-tune + 3 from-scratch), not 20. The entries
"2026-10-24: Phase 0 full run", "2026-11-19: ORCA access and specs" and
"2026-11-19: Phase 0 reproduced on ORCA" say 20.

### 2026-11-19: Seed-noise results (seed 1 vs seed 0, both on ORCA)

**Run:** `orca/seed_noise.sbatch`, seed 1, one A30, **17 min**. Results are on
ORCA in `~/lcm-runs/phase0_seed1/` and aren't copied to the laptop yet.
Data check: matches the laptop reference (10 files bit-identical, 6 within
rounding).

**How much results move from training chance alone (CRPS)**

| Run type | Changes (seed 1 vs seed 0) |
|---|---|
| Specialists (8) | **small**: median ~0.6%, max 7.3% (neurons) |
| Joint model (8 systems) | mostly ≤ 3.7%, but **neurons −18.7%, epidemic −9.0%** |
| Never-seen-it (zero-shot, 3) | **large**: epidemic +25.4%, neurons +12.8%, Kuramoto +2.7% |
| Fine-tune, 64 episodes (3) | moderate: up to 7.7% |
| From scratch, 64 episodes (3) | **very large**: epidemic −52.6%, neurons −40.3%, Kuramoto −4.0% |
| Odds error | main runs median 0.005 (max 0.040); few-shot max 0.142 |

Summary from `compare.py --noise`: main runs median 2.7%, max 25.4%;
few-shot median 5.8%, max 52.6%.

**What happens to the Phase 0 findings**
- **"Specialists beat the joint model": holds, except on epidemics, which
  is a tie.** Epidemic went from 0.041 vs 0.040 (specialist wins) to 0.037
  vs 0.037 (joint counted as winning), so the headline flipped from 0/8 to
  1/8. Every other gap is far larger than the noise (e.g. Kuramoto 0.287 vs
  0.145).
- **"Zero-shot beats the baseline on neurons only": holds.** With seed 1,
  neurons 0.728 vs baseline 1.097 still beats it; epidemic and Kuramoto
  still lose.
- **"Pretraining helps few-shot learning (3/3)": the direction holds, but
  the size was overstated.** Scratch-to-fine-tune error ratio:

  | System | Seed 0 | Seed 1 |
  |---|---|---|
  | neurons | 2.3× | 1.3× |
  | epidemic | 2.5× | **1.1×** |
  | Kuramoto | 1.3× | 1.3× |

  Seed 0's from-scratch models were unlucky. The headline "epidemic error
  0.10 vs 0.25" (2026-10-24 entry and README) overstates the effect. A fair
  statement is "pretrained models were better in all 6 seed–system pairs,
  by 1.1–2.5×".

**Consequences for later experiments**
- The `compare.py` reproduction rule ("main runs within 10%") only suits
  same-seed comparisons, like laptop vs ORCA. Across seeds, the joint and
  zero-shot runs moved up to 25%, so it would wrongly report FAIL.
- **The scale-up must use several seeds per configuration**, and should
  compare averages against this spread. One run per model can't separate a
  real gain below ~20% from luck for joint or zero-shot models.
- Few-shot conclusions need **more seeds or more few-shot episodes** than
  64; from-scratch runs on 64 episodes vary by up to 50% between seeds.
- One seed pair is a rough estimate. Seeds 2–4 (4 × 17 min, free; they can
  run at the same time as separate jobs) would give a proper spread.

### 2026-11-19: Seeds 2–4 submitted; five-seed summary script

**Decision:** run seeds 2–4 now (PJ, after discussing the value).
- *Why:* one seed pair can't show which seed was the outlier. For example,
  from-scratch epidemic scored 0.247 with seed 0 and 0.117 with seed 1, so
  pretraining's epidemic advantage could be 1.1× or 2.5×.
- The scale-up will need the current 1.08M model averaged over several seeds
  as its baseline anyway.
- Cost: 3 parallel ORCA jobs, ~17 min, free.

**Built:** `orca/noise_summary.py` takes any number of seed folders and
reports:
- per run: mean CRPS, standard deviation, relative spread (cv) and range;
- per run type: typical and worst cv;
- per headline finding: how many seeds it holds in, flagging seed-dependent
  ones;
- pretraining's few-shot advantage (from-scratch / fine-tuned error) per
  seed;
- a rough number of seeds per model to detect a 10% or 20% improvement.
  - Rule: n ≥ 8·cv²/effect², using the **worst** cv of each run type so it
    covers every run.
  - An earlier draft used the typical cv. It said 2 seeds were enough for
    the joint model despite joint-neurons swinging ~15%, so it was changed.
- The script rejects folders with duplicate seeds.

**Tested**
- `tests/test_noise_summary.py` (5 tests, synthetic seed folders with known
  answers): per-run spread, seed-dependent findings, the advantage ratio,
  duplicate-seed rejection, and the seeds-needed rule. 30/30 tests pass.
- Preview on seed 0 plus seed 1 rebuilt from the pasted log: the epidemic
  joint-vs-specialist result is correctly flagged as seed-dependent.
  - Rough seeds-needed figures from that pair, for a 20% improvement: joint
    5, zero-shot 6, from-scratch 51 (that run type is unusable at 64
    episodes).
  - These need the five-seed data to be trusted.

**Pending:** seeds 2–4 on ORCA, then `noise_summary.py` across all five
(`orca/README.md` step 8).
