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
