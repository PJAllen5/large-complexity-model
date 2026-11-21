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

| Setting | Phase 0 (run) | Scale-up Stage A (decided 2026-11-20) | Phase 1 start (planned) |
|---|---|---|---|
| **Parameters** | **1.08M** | **8.5M and 28M** (plus the 1.08M baseline) | **~28–50M** |
| Width / layers / heads | 128 / 4 / 4 | 256 / 8 / 8 and 384 / 12 / 8 | 384–512 / 12 / 8 |
| Mixture components (bell curves per forecast) | 4 | 4 | 8 |
| Systems trained on | 8 simulated | **same 8** (more systems = Stage B, later) | simulated + 2–3 real datasets |
| Episodes per system | 2,000 | 2,000; raise if validation shows overfitting | varies with the dataset |
| Validation episodes per system | none | 256 (`--n-val`) | yes |
| Few-shot episodes | 64 | **256** | to be decided |
| Seeds per configuration | 1 (5 measured later) | **3** | 3+ |
| Max entities / features per system | 12 / 4 | 12 / 4 | needs lifting to hundreds of entities |
| Time steps: history shown / forecast | 48 / 16 | 48 / 16 | to be decided per dataset |
| Patch (time steps per token) | 4 | 4 | 4–8 |
| Training steps per model | 5,000 | set from the pilot | ~100,000+ |
| Batch size (episodes per step) | 32 | set from the pilot (32 or 128) | ~256 |
| Learning rate | 3e-4 (1e-4 for fine-tuning) | 3e-4 | tuned |
| Precision | fp32 | fp32, tf32 or bf16, set from the pilot's accuracy check | tbd |
| Hardware | MacBook Air M3; rerun on ORCA A30 | ORCA A30 / L40S, at most ~8 GPUs at once (self-imposed; Slurm sets no per-user limit, see 2026-11-20 limits entry) | ORCA or rented A100/H100 |
| Time / cost | 82 min laptop, 17 min A30 / free | est. 50–150 GPU-h / free | 100s–1000s GPU-h |

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

### 2026-11-19: Five-seed noise results (seeds 0–4, ORCA)

**Runs:** seed 0 (`results_orca/`) and seeds 1–4 (`~/lcm-runs/phase0_seed*`
on ORCA), each a full Phase 0 run on one A30, ~17 min. Seeds 2–4 ran in
parallel. Summarised with `orca/noise_summary.py`. The seed 1–4 result
files are still only on ORCA.

**Spread across seeds** (cv = standard deviation / mean of CRPS)

| Run type | Typical cv | Worst cv | Seeds for a 20% / 10% improvement* |
|---|---|---|---|
| Specialist | 1.7% | 10.0% (epidemic) | 3 / 9 |
| Joint | 2.5% | 7.7% (neurons) | 2 / 5 |
| Zero-shot | 8.3% | 9.5% (epidemic) | 2 / 8 |
| Fine-tune, 64 episodes | 6.2% | 12.9% (epidemic) | 4 / 14 |
| From scratch, 64 episodes | **32.8%** | **45.3%** (neurons) | **42 / 165** |

\*Rough rule n ≥ 8·cv²/effect², using the worst cv.

- Seed 1's large moves (joint neurons −18.7%, zero-shot epidemic +25.4%)
  were **at the extreme**. Typical noise for joint and zero-shot runs is
  2–9%, not 20–25%.
- From-scratch training on 64 episodes is wildly unstable. Neurons ranged
  0.211–0.705 across seeds.

**Phase 0 findings across 5 seeds**
- **Specialists beat the joint model on 7 of 8 systems in 5 of 5 seeds.**
  Epidemic is a **tie**: mean 0.039 joint vs 0.037 specialist, joint ahead
  in 2 of 5 seeds.
- **Zero-shot beats the baseline on neurons in 5 of 5 seeds** (mean 0.639 vs
  1.097). Kuramoto in 1 of 5 (mean 0.600 vs 0.540, so it loses on average).
  Epidemic in 0 of 5.
- **Pretraining helps few-shot learning in 14 of 15 seed–system pairs.** The
  exception is neurons with seed 3: 0.99×, a tie.

  | System | Mean advantage (from scratch / fine-tuned) | Per seed |
  |---|---|---|
  | neurons | 1.90× | 0.99–3.26× |
  | epidemic | 1.77× | 1.14–2.47× |
  | Kuramoto | 1.32× | 1.26–1.38× |

  The seed-0 figure for epidemic (2.47×) was the highest of the five, so
  the original "0.10 vs 0.25" claim was the luckiest case.

**README updated** to these five-seed figures. The odds claims are still
from a single run.

**Decisions for the scale-up**
- **At least 3 seeds per configuration** (5 to detect ~10% gains), including
  the current 1.08M model as the baseline. These five runs already provide
  that baseline.
- **Few-shot comparisons need more than 64 episodes.** From-scratch runs at
  that size would need ~40 seeds to show a 20% difference. Options: 256+
  episodes, or comparing against the multi-seed average with a wide margin.

### 2026-11-20: Scale-up plan (Stage A) decided

**Question:** does more model capacity close the joint model's gap with the
specialists? Phase 0 put the gap down to the model being too small. This
tests that.

**Baseline:** the joint/specialist CRPS ratio, averaged over 5 seeds, is
**1.34** (geometric mean over the 8 systems). Per system it ranges from
1.05 (epidemic) to 1.89 (Kuramoto).

**Success criteria, fixed before running**
- *Gap shrinks:* the ratio falls by ≥ 0.10 at 8.5M. Joint and specialist
  runs each vary ~2% between seeds, so this is well beyond noise.
- *Gap closes:* the ratio is ≤ 1.05 (a tie within noise) at some size.
- *Stop and rethink:* the ratio is still > 1.2 at 28M. Then capacity isn't
  the bottleneck: investigate the data format, the training method or the
  variety of systems before scaling further.
- *Secondary:* zero-shot beats the baseline on more than 1 of 3 held-out
  systems, and pretraining's few-shot advantage holds or grows.

**Decisions (PJ accepted all the recommendations)**
1. **Model size first (Stage A); more systems later (Stage B).** *Why:*
   changing both at once couldn't say which one helped.
2. **Sizes 8.5M and 28M**; 50M skipped (about twice the cost).
3. **Specialists scaled to the same size**, for a fair comparison.
4. **2,000 episodes per system**, raised only if validation NLL shows
   overfitting. Data is free from the simulators.
5. **Few-shot tests use 256 episodes, not 64.** Five seeds showed
   from-scratch at 64 is too noisy (cv 33–45%).
6. **3 seeds per configuration**, compared on averages.
7. **At most ~8 ORCA GPUs at once** (shared resource).

**Build order**
- *PR A (this one):* size and training options, validation, faster-math
  option, pilot.
- *PR B:* one Slurm job per model with dependencies, and a scaling report.
- Then the runs.

### 2026-11-20: Scale-up PR A: options, validation, precision, pilot

**Built**
- **`experiment.py` options:** `--d --layers --heads --steps --batch --lr
  --n-train --n-val --few-shot-eps --few-shot-steps --precision`. Defaults
  reproduce Phase 0 exactly.
- **`config.json` per output folder.** The experiment **refuses** to
  continue in a folder whose saved settings differ. *Why:* runs are cached
  by name, so a second model size in the same folder would silently reuse
  the first size's results.
- **Validation split** (`*_val.npz`, its own seed, never overlaps
  train/test).
  - `train()` logs the validation NLL alongside the training NLL, so
    overfitting shows as a rising validation NLL.
  - `build_datasets` now rebuilds a cached file whose size differs from
    what's requested.
- **Precision option** (`lcm/train.py`):
  - `fp32` is exact (the default).
  - `tf32` uses NVIDIA tensor cores for matrix multiplies.
  - `bf16` runs the transformer blocks under autocast.
  - The **forecast head always stays fp32**, so the forecast means and
    widths aren't rounded. The attention mask is cast to match bf16 inputs.
- **`report.py`** loads the joint model at the configured size.
- **`orca/pilot.py` + `orca/pilot.sbatch`:**
  - Part 1: step time and peak GPU memory for 1.08M / 8.5M / 28M × fp32 /
    tf32 / bf16 × batch 32 / 128, with projected GPU-hours.
  - Part 2: the 8.5M model trained 1,500 steps in each precision from
    identical starts; validation NLL compared.
  - Run once on an A30 and once on an L40S.

**Tested on the laptop**
- 36/36 tests pass. New in `tests/test_scaleup.py`:
  - 8.5M / 28M parameter counts and forward pass;
  - bf16 autocast on CPU keeps head outputs fp32, with the loss within 0.1
    of exact;
  - **validation leaves training identical** (same seed → identical
    weights);
  - an unknown precision is rejected;
  - the validation split is separate from train, and is rebuilt on a size
    change.
- End-to-end `experiment.py --quick` with a small model (d=64), `--n-val
  16` and short training:
  - all 18 runs completed, with validation NLL logged;
  - the report was written at that size;
  - `config.json` was saved;
  - rerunning with a different width was refused;
  - rerunning with the same settings resumed 18/18 from cache.
- `orca/pilot.py --tiny` runs on the laptop (fp32 only).

**Not tested:** tf32 / bf16 on a real NVIDIA GPU, and actual speeds and
memory. That's what the pilot measures on ORCA.

**Open, to be set from the pilot:** training steps, batch size and
precision for the scale-up.

### 2026-11-20: Scale-up pilot results, and settings decided

**Runs:** `orca/pilot.sbatch` on an A30 (job 271955, 4 min) and an L40S
(job 271956, 2 min). torch 2.15.0+cu132.

**Accuracy (8.5M, 1,500 steps, identical start and batches)**

| Precision | A30 | L40S |
|---|---|---|
| fp32 | −0.999 | −1.003 |
| tf32 | −1.001 | −1.002 |
| bf16 | −1.004 | −1.006 |

Validation NLL; lower is better. Every precision is within ±0.005 of fp32,
far below normal run-to-run variation. **bf16 ran without errors on real
NVIDIA hardware** (first test). *Caveat:* this was a 1,500-step run, not
a full-length one.

**Speed** (seconds per step, batch 32)

| Size | A30 fp32 | A30 bf16 | L40S fp32 | L40S bf16 |
|---|---|---|---|---|
| 1.08M | 0.017 | 0.009 | 0.008 | 0.009 |
| 8.5M | 0.065 | 0.027 | 0.024 | 0.016 |
| 28M | 0.171 | 0.058 | 0.063 | 0.026 |

- bf16 is 2.4–2.9× faster than fp32 on the A30 for 8.5M and 28M.
- The L40S is ~2× faster than the A30 for 28M in bf16.
- **Memory:** at most 10.8 GB (28M, batch 128, fp32), well within the A30's
  24 GB and the L40S's 48 GB.
- **Projected cost:** Stage A comes to ~15–30 GPU-h, down from the earlier
  50–150 GPU-h estimate.

**Decisions (PJ accepted all the recommendations)**
1. **bf16 for all Stage A runs, including a fresh 1.08M baseline**, so every
   size is compared under identical settings. Phase 0's five fp32 seeds
   remain as reference only.
2. **Batch 32 and learning rate 3e-4, as in Phase 0.** Stage A changes only
   model size.
3. **20,000 training steps for every size**, including the 1.08M baseline,
   with 3 seeds. *Why:* at Phase 0's 5,000 steps, the bigger models might
   be undertrained and wrongly show that size doesn't help.
4. **An overfitting check first.** At 20k steps, a specialist sees its 2,000
   episodes ~320 times. The check trains 8.5M and 28M specialists on two
   systems with validation on, before the full launch.
5. **GPUs:** 28M on L40S, 1.08M and 8.5M on A30, at most 8 GPUs at once.
6. **Few-shot:** 256 episodes, 2,000 steps. That's the same ~250 passes over
   the data as Phase 0's 64 episodes × 500 steps.

### 2026-11-20: Scale-up PR B: one Slurm job per model

**Built**
- **`experiment.py` cluster modes:**
  - `--only <tasks>` runs single tasks: `joint`, `specialist_<system>`,
    or `transfer_<system>`, which runs never-seen-it, then fine-tune, then
    from-scratch in one job, since fine-tuning needs the never-seen-it
    model first.
  - `--data-dir` points at a shared data folder.
  - `--prepare-data-only` simulates the data and stops.
  - `--report-only` writes the report once every run exists, and refuses
    otherwise.
  - Training now logs 20 progress points instead of 5, so validation
    curves are readable.
- **Atomic writes:** data files, run results, checkpoints and
  `config.json` are written to a temporary name, then renamed. *Why:*
  parallel jobs share the data folder and must never read a half-written
  file.
- **Stage A scripts:**
  - **`orca/stageA_settings.sh`** holds the single copy of the Stage A
    settings (bf16, 20k steps, batch 32, lr 3e-4, 256 validation and 256
    few-shot episodes, 2k few-shot steps, seeds 0–2) and the size → shape
    map.
  - **`orca/stageA_submit.sh [--check]`** submits:
    - a CPU job for the shared data;
    - an A30 array for 1.08M and 8.5M (at most 6 at once);
    - an L40S array for 28M (at most 2 at once);
    - on a full run, a report job that waits for every task to succeed.

    It **leaves out tasks that already have results**, so rerunning it
    after a failure only resubmits unfinished work. `--check` runs the 4
    overfitting-check specialists in the same folders, so the full run
    reuses them.
  - **`orca/stageA_task.sbatch`**, **`stageA_prep.sbatch`** and
    **`stageA_reports.sbatch`** are the job scripts. Plain scripts are used
    instead of inline `--wrap` commands, which Slurm runs under `/bin/sh`.
  - **`orca/val_curves.py`** shows validation curves and flags overfitting:
    a rise of more than 0.05 NLL after the best point (a judgement call).

**Tested on the laptop**
- 42/42 tests pass. New in `tests/test_stageA_tools.py`: overfitting,
  still-improving and small-wobble verdicts; runs without validation are
  skipped; the size → shape map; transfer tasks produce 3 runs.
- **Simulated end-to-end submit**, in a throwaway repo copy with a
  stand-in `sbatch` that runs jobs immediately, tiny models, 6 steps and
  1 seed:
  - `--check`: 4 tasks, all succeeded.
  - Full run: only the remaining **32 of 36** tasks were submitted (the
    check's 4 were reused), all succeeded, and the report job wrote reports
    for all 3 size folders (18 runs each, bf16 recorded in `config.json`).
  - Resubmit: **0** unfinished tasks; only the reports were regenerated.
- **Bugs found by the simulation (fixed):**
  - The first attempt failed every task. The test copy had no git history,
    so the job's `git log` line (which only prints the code version)
    stopped it under `set -e`. That line is now non-fatal in the task and
    pilot jobs.
  - `val_curves.py` printed steps under 1,000 as "0k", and could repeat
    the last point.

**Not tested:** real Slurm (dependencies, array throttling, `--export`),
real GPUs and real run times. The `--check` launch on ORCA is the first
real test.

### 2026-11-20: Overfitting check: Lorenz96 specialists memorise; more data and early stopping

**Run:** `bash orca/stageA_submit.sh --check`. Data job 271957 (24 s, CPU);
A30 array 271958 (8.5M) and L40S array 271959 (28M); seed-0 specialists for
epidemic and Lorenz96; 2,000 training episodes, 20k steps, bf16. **This was
the first real test of the Stage A Slurm setup, and it worked:** the
dependencies, both arrays and the shared data all behaved as designed.
Results are in `~/lcm-runs/stageA/` on ORCA and are kept as evidence.

**Result** (validation NLL; lower is better)

| Specialist | Best val | Val at 20k | Training NLL at 20k |
|---|---|---|---|
| Epidemic 8.5M | still improving at 20k: −3.96 | | |
| Epidemic 28M | still improving at 20k: −3.995 | | |
| **Lorenz96 8.5M** | −2.14 at 10k | −1.06 | −2.88, still falling |
| **Lorenz96 28M** | −2.06 at 8k | **+5.82** | −3.40, still falling |

**Diagnosis: overfitting, not instability.** The training NLL falls
smoothly to the end while the validation NLL turns up after ~8–10k steps,
which is about 130–160 passes over the 2,000 episodes. The bigger model
memorises more.

**Decisions (PJ accepted all three)**
1. **20,000 training episodes per system** (10×), so 20k steps is about 32
   passes per specialist. This is the planned decision rule ("raise
   episodes if validation shows overfitting"). Data is free from the
   simulators.
2. **Keep the best-validation checkpoint (early stopping).** `train()`
   restores the weights from the logged step with the lowest validation
   NLL, and each run records `best_step`. It's applied identically to every
   size.
3. **Rerun the check with the new settings** before the full launch, in a
   fresh folder: `~/lcm-runs/stageA_n20k/`.

**Note:** Phase 0 had no validation, so some of its runs (80 passes over
2,000 episodes) may have overfitted too. Stage A reruns the 1.08M baseline
under the new settings, so the size comparison is fair either way.

**Built:** `train(..., keep_best=True)`, `best_step` in run results,
`--n-train 20000` and the new folder in `orca/stageA_settings.sh`, and
updated paths in the guide.

**Tested**
- 43/43 tests pass.
- The new early-stopping test feeds a dipping-then-rising validation
  sequence and checks the model ends with the dip's weights. A version of
  the test where the best point happened to be the last step was rejected
  as trivial. The final test **fails when early stopping is switched off**.
- A quick end-to-end run records `best_step` and marks `*best` in the log.

**Not tested:** 20,000-episode data on ORCA (generation time, memory). The
rerun of `--check` is that test.

### 2026-11-20: Overfitting fixed; full Stage A launched; scaling report (PR C)

**Overfitting re-check** (20,000 episodes, best-validation checkpoint;
`~/lcm-runs/stageA_n20k`): all four validation curves **improve to the end
of training**; none rise. Final validation NLL (lower is better):

| Specialist | Before (2,000 episodes) | After (20,000 episodes) |
|---|---|---|
| Lorenz96 8.5M | −1.06 (best −2.14) | **−2.43** |
| Lorenz96 28M | +5.82 (best −2.06) | **−2.46** |
| Epidemic 8.5M | −3.96 | −3.99 |
| Epidemic 28M | −4.00 | −4.05 |

- More data didn't only stop the collapse: both Lorenz96 models now beat
  their earlier *best* points.
- An early hint, not yet a result: 28M edged out 8.5M on both systems.

**Full Stage A launched**, 2026-11-20:
- Data job 272190; A30 array 272191 (70 tasks); L40S array 272192 (34
  tasks); report job 272193. That's 104 tasks; the 4 check runs were reused.
- After submitting, the throttles were changed with `scontrol update ...
  ArrayTaskThrottle` to L40S 3 and A30 5. *Why:* the 28M tasks on L40S were
  the bottleneck (~4 h at 2 at once; ~2.5 h at 3).
- Total stays at 8 GPUs. Briefly 9 were in use, because 6 A30 tasks had
  started before the change; Slurm doesn't stop running tasks.

**Built: `orca/scaling_report.py`** (PR C)
- **Gap** = joint / specialist CRPS, geometric mean over the 8 systems,
  per seed. Reported as mean ± sd per size, plus per-system gaps.
- **Verdicts on the pre-registered criteria**:
  - shrinks: 8.5M ≥ 0.10 below 1.08M, with 2× the seed noise shown
    alongside;
  - closes: ≤ 1.05 at some size;
  - stop: > 1.2 at 28M.
- **Decision: the baseline for the verdicts is the Stage A 1.08M run**
  (same settings), not Phase 0's 1.34 (different precision, steps and data
  size). 1.34 is shown for reference only.
- **Secondary:** zero-shot wins per held-out system; pretraining's few-shot
  advantage; the number of runs whose validation rose after their best
  point.
- **Output:** `SCALING.md` and `scaling.png` (gap vs size; seed dots, mean
  line, the 1.05 and 1.2 lines). It handles partial results: unfinished
  seeds are left out, and verdicts that need missing sizes say "NOT YET".
- `stageA_reports.sbatch` now also runs it. The already-queued report job
  272193 keeps the old script, because Slurm copies scripts at submission.

**Tested**
- 50/50 tests pass. New in `tests/test_scaling_report.py`, on synthetic
  results with exact known gaps:
  - the gap computation;
  - a success case (shrinks, closes at 28M, no stop);
  - a stop case;
  - partial results giving "NOT YET";
  - a missing run leaving that seed out;
  - zero-shot, few-shot and overfitting counts;
  - the markdown and chart being written.
- Ran on the earlier simulated Stage A (real result files from tiny
  models): it reads them correctly.
- Chart checked visually; the line labels were moved so they don't
  overlap seed points.

**Not tested:** the real Stage A results, which are still running.

### 2026-11-20: Stage A result: bigger models did not close the gap (stop criterion met)

**Run:** full Stage A on ORCA. 104 tasks (plus 4 reused from the check):
A30 array 272191 (1.08M and 8.5M), L40S array 272192 (28M), throttled to 5
+ 3 GPUs; report job 272193. **All completed; none failed.** Took ~2.5 h,
free. Settings: bf16, 20k steps, batch 32, 20,000 episodes per system,
best-validation checkpoint, 3 seeds. Results:
[results_stageA/SCALING.md](results_stageA/SCALING.md); the run files are
copied to `results_stageA/` (data and checkpoints stay on ORCA).

**Gap = joint / specialist CRPS, geometric mean over 8 systems**

| Size | Gap mean (sd) | Per seed |
|---|---|---|
| 1.08M | **1.449** (0.016) | 1.463, 1.431, 1.453 |
| 8.5M | **1.540** (0.022) | 1.558, 1.545, 1.516 |
| 28M | **1.517** (0.017) | 1.508, 1.536, 1.506 |

**Verdicts on the pre-registered criteria**
- Gap shrinks by ≥ 0.10 at 8.5M: **NO.** It *grew* by 0.091, about 3× the
  seed noise.
- Gap closes (≤ 1.05): **NO.**
- Stop (> 1.2 at 28M): **YES.** "Capacity is the bottleneck" is not
  supported, so we investigate before scaling further.

**Per system: two opposite stories**

| | Systems | Gap, 1.08M → 28M |
|---|---|---|
| Gap shrinks | Kuramoto, gene network, neurons, epidemic | 1.54 → 1.39, 1.15 → 1.06, 1.25 → 1.16, 1.25 → 1.13 |
| Gap grows | **Lorenz96** | **3.57 → 5.17 (8.5M) → 4.07** |
| Gap grows | **n-body** | **1.38 → 1.72 → 2.26** |
| Gap grows slightly | predator–prey, boids | 1.18 → 1.26, 1.20 → 1.25 |

**Secondary results**
- **Zero-shot:** unchanged at every size (neurons 3/3; epidemic and
  Kuramoto 0/3).
- **Few-shot pretraining advantage:** shrinks for neurons (2.40× → 1.18×),
  holds for epidemic (1.70× → 1.89×), negligible for Kuramoto (~1.05×).
- **Overfitting:** 0 / 0 / 3 of 54 runs per size. The best checkpoint
  covered those 3.
- **Relative to Phase 0:** the 1.08M gap rose from 1.34 to 1.45 under the
  Stage A settings. The specialists gained much more from 10× data and 4×
  steps (e.g. Lorenz96 specialist 0.273 → 0.085) than the joint model did
  (0.349 → 0.304).

**Diagnosis from the existing results (no new compute)**
1. **The joint model does improve with size, just less than the
   specialists.** Geometric-mean CRPS vs 1.08M:

   | | 8.5M | 28M |
   |---|---|---|
   | Joint | ×0.835 | ×0.764 |
   | Specialists | ×0.786 | ×0.730 |

   Lorenz96: joint 0.304 → 0.181, specialist 0.085 → 0.044. n-body: joint
   0.262 → 0.189, specialist 0.189 → 0.084.
2. **The joint models were still improving when training stopped**, more
   so at bigger sizes. Best validation was at the final step in 9/9 joint
   runs. Validation NLL change over the last 5k steps:

   | Size | Joint | Specialists (mean) |
   |---|---|---|
   | 1.08M | −0.085 | −0.065 |
   | 8.5M | −0.145 | −0.092 |
   | 28M | −0.180 | −0.138 |

   *Caveat:* the cosine learning-rate schedule anneals to 0 at the end,
   which produces some late improvement in every run. The schedule is
   identical for both, so the comparison holds.

**Interpretation.** The leading explanation is that **the joint model is
undertrained relative to the specialists**. With equal steps, it gets about
1/8 of the per-system training (~2,500 steps each), and bigger models need
more training. This is consistent with the gap growing with size. Not yet
ruled out: interference between systems, especially the precise continuous
dynamics of Lorenz96 and n-body, and a data format that serves continuous
physics poorly.

**Proposed next test (not yet decided):** train the joint model **8×
longer (160k steps)**, so each system gets the same exposure as a
specialist, at all 3 sizes × 3 seeds. Compare it with the existing 20k-step
specialists.
- Estimated ~9–12 GPU-h on ORCA.
- If the gap then shrinks with size, the bottleneck was training budget,
  not capacity.
- If Lorenz96 and n-body still lag, look at interference or the data
  format.

### 2026-11-20: Longer joint training test set up (8× steps)

**Decision (PJ):** run the proposed test. The joint model is retrained for
**160,000 steps** (8 systems × 20k, so each system gets a specialist's
worth of training) at all 3 sizes × 3 seeds: 9 runs. It's compared with the
existing 20k-step Stage A specialists. Everything else is unchanged (bf16,
batch 32, lr 3e-4 with the cosine schedule stretched over 160k steps,
20,000 episodes, best-validation checkpoint).

**What the answers would mean**
- **The gap now shrinks with size** (the same pre-registered criteria
  apply, judged against the 160k-step 1.08M joint model): the bottleneck
  was the training budget, not capacity.
- **Lorenz96 and n-body still lag:** look at interference between systems,
  or the data format, next.
- **The gap doesn't move:** undertraining wasn't the main cause.

**Built**
- **`orca/stageA_settings.sh`:** `STAGEA_STEPS` and `STAGEA_DATA` can be
  overridden, like `STAGEA_ROOT`. Defaults are unchanged.
- **`orca/stageA_submit.sh --joint-long`:**
  - joint tasks only;
  - 160k steps;
  - results in `~/lcm-runs/stageA_joint160k`;
  - **reuses** the Stage A data folder (identical data);
  - a 3.5 h task time limit, against ~70–80 min expected for 8.5M on an
    A30 and 28M on an L40S.
- **Default GPU limits are now 5 A30 + 3 L40S** (were 6 + 2), matching the
  manual rebalance that sped up Stage A.
- **`orca/scaling_report.py --specialists-from <folder>`:** joint runs come
  from one run; specialists and transfer runs from another. A "Gap with the
  reference run's joint" column shows the 20k-step gaps side by side.

**Tested**
- 52/52 tests pass. New: settings overrides, and a combined report on
  synthetic folders with known gaps.
- **Simulated `--joint-long` submit** (stand-in Slurm, tiny models, 1
  seed):
  - only the 3 joint tasks were submitted, with the step override reaching
    every job, the 5 + 3 limits, and the shared data folder;
  - no failures;
  - the resubmit found nothing to do;
  - the combined report read both runs correctly.
- The normal Stage A report is unchanged.

**Not tested:** the real 160k-step runs (time, and whether 3.5 h is enough
on busy nodes).

**Estimated cost:** ~9 GPU-h; ~1.5–2 h of waiting.

### 2026-11-20: Longer joint training result: the joint model now beats the specialists

**Run:** 9 joint models trained for 160,000 steps (1.08M / 8.5M / 28M × seeds
0–2) on ORCA. Jobs: data 272978, A30 array 272979 (1.08M and 8.5M), L40S
array 272980 (28M). All 9 COMPLETED with no failures or timeouts (checked
with `sacct`). The 1.08M runs took ~29–31 min each on an A30. The other run
times weren't recorded here. Free (ORCA). Report:
[results_stageA_joint160k/SCALING.md](results_stageA_joint160k/SCALING.md)
(generated on ORCA with `--specialists-from ~/lcm-runs/stageA_n20k`, so the
specialists are the 20k-step Stage A ones).

**Headline: gap = joint / specialist CRPS, geometric mean over 8 systems**
(below 1.0 = the joint model is better)

| Size | Gap at 20k steps | Gap at 160k steps (sd over 3 seeds) |
|---|---|---|
| 1.08M | 1.449 | **0.958** (0.007) |
| 8.5M | 1.540 | **0.917** (0.013) |
| 28M | 1.517 | **0.919** (0.013) |

**Pre-registered checks** (judged on the 160k-step runs)
- Gap shrinks (8.5M ≥ 0.10 below 1.08M): **NO**, 0.958 → 0.917 (a 0.040
  drop, about 2× the seed noise). This check was written for a gap well
  above 1. It doesn't fit once the gap has already closed at 1.08M.
- Gap closes (≤ 1.05 at some size): **YES, at all three sizes.**
- Stop (> 1.2 at 28M): **NO.**

**Per system** (gap at 1.08M / 8.5M / 28M)
- **Lorenz96:** 1.29 / 1.01 / 0.97. It was 3.57 / 5.17 / 4.07 at 20k.
- **n-body:** 0.88 / 0.67 / 0.70. It was 1.38 / 1.72 / 2.26 at 20k.
- **Epidemic:** 0.71 / 0.83 / 0.87.
- **The rest are between 0.94 and 1.03.**
- **Both systems that got worse with size at 20k now improve with size.**

**Validation curves** (`orca/val_curves.py`)
- All 9 still had their best validation NLL at the final step (none
  overfitting).
- Final validation NLL (negative log-likelihood of held-out episodes; lower
  is better): 1.08M −2.20, 8.5M −2.44, 28M −2.50. Seeds agree within
  0.005.
- **On NLL, 28M is clearly better than 8.5M**, even though their CRPS gaps
  are tied.
- The learning rate follows a cosine decay to zero, so late improvements
  are always small. The curves still slope down, though (−0.03 over the
  last 24k steps).

**Conclusion**
- **The undertraining hypothesis is supported.** With equal training per
  system (160k/8 = 20k steps, the same as a specialist), one shared model
  beats 8 specialists by about 8% at 8.5M and 28M.
- **Main caveat:**
  - I checked the 72 Stage A specialists (20k steps) in `results_stageA/`
    afterwards, and all 72 had their best validation at the final step, so
    they were still improving too.
  - This is a comparison at **equal per-system training**, not at
    convergence (where both sides are trained until they stop improving).
  - Longer-trained specialists could win back some or all of the gap.
- **The transfer table in the report** is reused from the 20k-step
  reference run, so it isn't new evidence.

**Proposed next test (not yet decided):** train specialists for 160k steps
(8 systems × 3 sizes × 3 seeds = 72 runs). This tests whether the joint
model's win holds when the specialists also get 8× more training.

### 2026-11-20: ORCA compute limits checked

**Why:** before planning bigger jobs (such as 72 specialist runs at 160k
steps), find out what ORCA actually allows. PJ ran the commands below on
ORCA and pasted the output.

**Results**
- **QOS** (`sacctmgr show qos`): only `normal`, with **no caps**. That
  means no limits on GPUs per user, running or queued jobs, or job length.
  Together with the earlier `sacctmgr show assoc` (no account limits),
  Slurm sets **no usage limits** for `pjallen5`.
- **Partitions** (`scontrol show partition`): no partition QOS, unlimited
  nodes. The partitions share nodes, so the GPU counts overlap:

| Partition | Time limit | GPUs | A30 | L40S |
|---|---|---|---|---|
| short | 4 h | 76 | 52 | 24 |
| normal | 1 day | 84 | 64 | 20 |
| long | 7 days | 36 | 36 | **0** |
| osg | 3 days | 24 | 24 | 0 (Open Science Grid) |

- **Priority** (`scontrol show config`): multifactor, but **every weight is
  0** (age, fair-share, job size, partition, QOS).
  - All jobs have equal priority: the queue runs in submission order, with
    backfill.
  - `sshare` shows no fair-share row for PJ, so past usage doesn't lower
    priority.
- **Disk** (`quota -s`): home quota **90 GB soft / 100 GB hard**. 15.6 GB
  used (data, venv, runs).
  - The shared `/home` filesystem is 84% full (1.6 TB free).
  - No `/scratch` workspaces allocated (`ws_list` empty).

**What it means**
- **Jobs over 1 day can't use L40S GPUs** (`long` has none). A 28M run
  longer than 24 h on an L40S would need checkpoint/resume chunks.
- **Tight `--time` limits start sooner via backfill.** The 3.5 h joint-long
  limit (for ~80 min tasks) could be about 2 h next time.
- **The real limits are the written usage policy and courtesy.** Keep the
  self-imposed cap of ~8 GPUs at once (5 A30 + 3 L40S) until PSU's policy
  is checked.
- **Disk:** ~74 GB of headroom before the soft quota. Delete or move old
  `data/` and `checkpoints/` folders under `~/lcm-runs` before large new
  datasets. For big data, use a `/scratch` workspace (`ws_allocate`;
  30-day leases, not backed up).

**Not checked:** PSU's written usage policy for ORCA (ask the admins or
read their usage page).

### 2026-11-20: 160k-step specialist test set up

**Decision (PJ):** run the proposed test. The 8 specialists are retrained
for **160,000 steps** at **1.08M and 8.5M** × seeds 0–2 (48 runs). They're
compared with the existing 160k-step joint models, so both sides get the
same training per system and 8× what Stage A gave them.
- **28M is left out for now:** it tied with 8.5M on the gap, and would add
  ~30 GPU-h. Add it if the 8.5M result is close to a threshold.
- **Everything else is unchanged:** Stage A data (shared folder), bf16,
  batch 32, lr 3e-4 with cosine decay over 160k steps, best-validation
  checkpoint.

**Pre-registered verdict** (gap = 160k joint / 160k specialist CRPS,
geometric mean over 8 systems; seed noise ≈ 0.01), judged per size, with
8.5M as the headline:
- **Joint still wins:** gap ≤ 0.97.
- **Tie:** 0.97 < gap ≤ 1.03. Sharing costs nothing, which still matters
  for transfer.
- **Specialists win:** gap > 1.03. The earlier win was mostly a
  training-length effect; look at interference between systems next.

**Built**
- **`orca/stageA_submit.sh --specialists-long`:**
  - specialist tasks only, 160k steps;
  - results in `~/lcm-runs/stageA_spec160k`, reusing the Stage A data;
  - 8.5M on A30s (same GPU type as its joint model), 1.08M on L40Ss;
  - 3 h default task limit.
- **`TASK_TIME` can be overridden** for any mode (e.g. `TASK_TIME=02:00:00`),
  so the limit can be tightened once the 8.5M run time is known.
- **Bug found and fixed while testing:** in the first version, the default
  2 h limit was set before the mode, so the longer per-mode limits were
  ignored. The joint-long run used the old script and wasn't affected.
- **`orca/convergence_report.py`:**
  - joint from `stageA_joint160k`, specialists from `stageA_spec160k`;
  - the verdict per size, the per-system gaps, the gap against the 20k
    specialists for reference;
  - how much each specialist improved from 20k to 160k steps;
  - overfitting / still-improving counts for the specialists;
  - writes `CONVERGENCE.md` into the specialists folder.

**Tested**
- **65/65 tests pass**, with 13 new ones:
  - report thresholds, gaps on synthetic folders, unfinished seeds, sizes
    without specialists, curve counts;
  - the submit mode with a stand-in `sbatch` (48 tasks split 24/24, the
    right sizes and GPU types, skipping finished runs, per-mode and
    overridden time limits).
- **Tiny end-to-end run on the laptop:** a real joint run, 8 long and 8
  short specialists (tiny model, 80 and 10 steps), then the report. It
  completed and the report rendered.
- **The report on real files** (20k specialists standing in for the 160k
  ones) reproduces the known 1.08M gap of 0.958.

**Not tested:** the real ORCA runs, or the actual time per 8.5M specialist
task on an A30.

**Estimated cost:** ~40 GPU-h (24 × ~1.25 h on A30, 24 × ~0.4 h on L40S),
about 6 h of waiting. Free.

### 2026-11-21: 160k-step specialist result: the specialists win (correction)

**Correction to the 2026-11-20 entry "Longer joint training result: the
joint model now beats the specialists":**
- **That headline was wrong.** The joint model beat specialists that were
  themselves undertrained (20k steps, all still improving).
- **With both sides trained for 160k steps, the specialists win at both
  sizes tested.**
- The caveat in that entry was the deciding factor. The 2026-11-20 entry
  is left as written.

**Run:**
- 48 specialist runs, 160k steps each: 8 systems × 1.08M and 8.5M × seeds
  0–2. All 48 COMPLETED.
- **8.5M:** A30 array 273515, 5 at a time, **1 h 14–15 min** per run.
- **1.08M:** planned for L40Ss (array 273516), but **all six L40S nodes
  were out of service** (orcaga02, 03, 05 and 06 down after failed
  reboots; 01 and 04 drained). The array was cancelled before any task
  started and resubmitted on A30s (array 273522, 3 at a time, 1 h limit):
  **26.7–27.1 min** per run. This matches where the 1.08M joint models
  ran.
- Time limit: `TASK_TIME=02:00:00`, set from the measured 1 h 17 min of
  the 8.5M joint runs.
- **~40.6 GPU-h in total**, about 6.5 h of waiting. Free.
- Report:
  [results_stageA_spec160k/CONVERGENCE.md](results_stageA_spec160k/CONVERGENCE.md).
  Reproduced on the laptop from the copied results with identical gaps.

**Headline:** gap = 160k joint / 160k specialist CRPS (geometric mean over
8 systems)

| Size | Gap vs 20k specialists (2026-11-20) | Gap vs 160k specialists (sd) | Pre-registered verdict |
|---|---|---|---|
| 1.08M | 0.958 | **1.289** (0.021) | Specialists win |
| 8.5M | 0.917 | **1.079** (0.005) | Specialists win |

**Per system at 8.5M**
- Near ties: predator-prey 1.01, gene networks 1.01, neurons 1.02, boids
  1.03, Lorenz96 1.03.
- Losses: Kuramoto 1.10, epidemic 1.11, **n-body 1.36**.
- At 1.08M every system loses, and Lorenz96 is worst (2.58).

**What the specialists gained from 8× training** (160k / 20k specialist
CRPS)
- n-body 0.49 at 8.5M; epidemic 0.63 / 0.75; Lorenz96 0.50 at 1.08M.
- The rest 0.78–1.00.

**Specialist validation curves**
- **1.08M:** 15 of 24 still improving at the end, 3 overfitting. n-body
  peaked at 64k–96k steps.
- **8.5M:** **18 of 24 overfitting**. The best checkpoints were at 40k–128k
  steps for every system except epidemic (160k). With 20,000 episodes of
  one system, an 8.5M specialist has learned all it can by about 50k–100k
  steps.
- **By contrast, all 9 joint models were still improving at 160k** (see the
  2026-11-20 entry).

**What it means**
- **Sharing one model across 8 systems costs accuracy on those systems at
  these sizes:** 29% at 1.08M, 8% at 8.5M. It hasn't been shown to help
  forecasting on systems the model was trained on.
- **The cost shrinks fast with size: 1.289 → 1.079.** That's a drop of
  0.21 with seed sd ≤ 0.02. Under fair training, this clears Stage A's
  pre-registered "gap shrinks" bar (≥ 0.10), which the undertrained Stage A
  runs missed. That bar was set for the Stage A comparison, so this is an
  observation, not a pre-registered pass.
- **The two sides are in different states at 8.5M:**
  - specialists are data-limited: they overfit, and more steps won't help;
  - the joint model is still improving.

  More training or a bigger joint model could narrow the gap further.
- **Undertraining hypothesis, revised:** the joint models *were*
  undertrained at 20k steps, but so were the specialists. Training both
  longer helped the specialists more at these sizes.
- **The case for one model now rests on two things:**
  - the size trend (does the gap reach ≤ 1.03 at 28M?);
  - transfer to new systems, where pretraining lowers few-shot error
    1.7–1.9× on epidemics and 1.2–2.4× on neurons (smaller at bigger
    sizes), and barely on Kuramoto (Stage A, 20k-step runs).

**Proposed next test (not yet decided):** 160k-step specialists at 28M
(24 runs), compared with the existing 28M joint models (160k).
- This tests whether the gap keeps falling (1.29 → 1.08 → ?).
- **It must run on A30s while the L40S nodes are down.** The 28M run time
  on an A30 is unknown and may exceed 2 h, so time one run first.

### 2026-11-21: 28M specialist test set up (on A30s)

**Decision (PJ):** run the proposed test. 8 specialists × **28M** × seeds
0–2 (24 runs), 160k steps each. They're compared with the existing 160k-step
28M joint models.
- **Question:** does the gap keep falling with size (1.289 at 1.08M →
  1.079 at 8.5M → ?).
- **Verdict:** the same pre-registered thresholds as the 1.08M / 8.5M
  runs: joint still wins ≤ 0.97, tie up to 1.03, specialists win above.
- **The size trend is read alongside it:** a 28M gap near 1.03 or below
  would mean size alone closes the gap at this data scale.

**Hardware:** all six L40S nodes were out of service on 2026-11-21, so this
runs on **A30s, 8 at a time**. That's the whole 8-GPU cap, since no L40S
can be used.
- From the pilot (0.058 s/step for 28M bf16 on an A30) and the 8.5M runs
  matching their pilot estimate, each run should take **~2.6–2.7 h**.
- Time limit 3.5 h (the `short` partition's maximum is 4 h).
- ~64 GPU-h, about 8 h of waiting.
- **Caveat:** the 28M joint models ran on L40Ss. The pilot found bf16
  accuracy differs by ≤ 0.005 NLL between the two GPU types, far below
  seed noise.

**Built:** `orca/stageA_submit.sh` takes environment overrides:
- `STAGEA_A30_SIZES` / `STAGEA_L40S_SIZES` (sizes per GPU type; empty =
  none);
- `A30_MAX` / `L40S_MAX` (tasks at once);
- an unknown size is refused before anything is submitted.
- Defaults are unchanged. Results go to the same
  `~/lcm-runs/stageA_spec160k` folder, so `orca/convergence_report.py`
  picks up 28M with no changes.

**Tested:**
- 67/67 tests pass. New: 28M on A30 only (24 tasks, 8 at a time, a 3.5 h
  limit, no L40S array), and refusing an unknown size.
- The existing submit tests confirm the defaults are unchanged.

**Not tested:** the real 28M run time on an A30.
