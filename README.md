# LCM: Large Complexity Model, Phase 0

An LLM learns patterns in text and predicts the next word. An LCM learns
patterns in **complex systems** and predicts **the odds of what happens next**.

Phase 0 tests the core idea on a laptop. Can **one** model, trained on many
very different systems, forecast all of them, give correct odds, and pick up
a system it has never seen?

## The eight training systems

| system | field | entities | what's observed |
|---|---|---|---|
| `predator_prey` | ecology | 6 species in a random food web | log abundance |
| `boids` | animal behaviour | 12 flocking birds | position, velocity |
| `kuramoto` | synchronisation | 12 coupled oscillators (fireflies, heart cells) | phase |
| `lorenz96` | climate / chaos | 10 points on a ring of atmosphere | state |
| `nbody` | astronomy | 6 bodies under gravity | position, velocity |
| `gene_reg` | cell biology | 8 genes switching each other on/off | expression |
| `neurons` | neuroscience | 10 coupled spiking neurons (FitzHugh–Nagumo) | voltage, recovery |
| `epidemic` | society | 10 cities linked by travel (stochastic SIR) | susceptible / infected / recovered |

Every system is random: each episode draws a new food web, coupling strength,
infection rate and so on, and all of them have noise. So "the future" is a
range of possibilities, not one answer.

## How it works

1. **Universal format** ([lcm/data.py](lcm/data.py)). Every system becomes
   *entities × time × features* plus a graph of who interacts with whom.
   Each episode is rescaled using only its past, so a galaxy and a gene look
   alike to the model.
2. **Tokens** ([lcm/model.py](lcm/model.py)). One token = 4 time steps of one
   entity. That is the system's "word".
3. **Transformer.** Each layer looks back along every entity's own history,
   then across entities at the same moment. A learned bonus nudges attention
   toward entities that interact.
4. **Probabilistic output.** For each token the model predicts a *mixture of
   bell curves* over the next 4 steps. Sampling step after step gives whole
   possible futures. Counting how often an outcome happens gives its odds.
5. **The model is never told which system it is looking at.** It has to work
   that out from the history, the way an LLM works out the topic from the prompt.

## What the experiment tests ([experiment.py](experiment.py))

- **Joint vs specialist.** One model on all 8 systems vs 8 single-system
  models, all with the same training budget.
- **Zero-shot.** Hide a system (neurons, epidemic or kuramoto) during
  training, then ask the model to forecast it.
- **Few-shot.** Give the model that never saw a system only 64 examples of
  it, and compare with a fresh model given the same 64.
- **Odds** ([lcm/evaluate.py](lcm/evaluate.py)). Run a system to some moment,
  then replay its future 64 times in the simulator to get the *true* odds that
  a quantity will rise. Compare with the model's odds.

Scores:
- **CRPS** grades the whole predicted distribution; lower is better.
- **Baseline** is "nothing changes", which repeats the last value.
- **90% coverage** is how often the truth falls in the model's 90% range; ideal is 90%.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python experiment.py --quick
```

```bash
.venv/bin/python experiment.py
```

The `--quick` run takes about 7 minutes. The full run takes about 1–1.5 hours
on an M3 MacBook and also works on CUDA or CPU. Results go to
`results/REPORT.md` with charts. Finished runs are cached, so an interrupted
experiment resumes where it stopped.

### Tests

```bash
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest -q
```

About 2 seconds on CPU. They check the data format, that normalisation never
uses future values, the parameter count, that predictions are causal in time,
that empty entity slots are ignored, sampling, and that a tiny model can learn.
GitHub runs them on every pull request.

## Add a new system

Subclass `Simulator` in [lcm/simulators.py](lcm/simulators.py) with `init`,
`step` and `obs`, put the interaction graph in `params["adj"]`, and add it to
`SIMULATORS`. Up to 12 entities and 4 features fit the current padding
(`NMAX`, `FMAX` in `lcm/data.py`). Nothing else changes.

## Results

Full numbers and charts are in [results/REPORT.md](results/REPORT.md). Every decision and test is recorded in [PROJECT_LOG.md](PROJECT_LOG.md).
Headline from the first full run (82 min on an M3 MacBook), with the last
three findings re-checked across 5 training seeds on ORCA (the odds figures
are from a single run):

- **The odds work.** Across all 8 systems, the joint model's odds are off by
  0.24 on average. An informed guess that knows each system's average odds is
  off by 0.40. The truth falls inside the model's 90% range 87% of the time.
- **Experience transfers.** Given only 64 examples of a system it never saw
  in training, the LCM beat a fresh model on all 3 held-out systems.
  Across 5 training seeds it won in 14 of 15 seed–system pairs (the other
  was a tie), with on average **1.3–1.9× lower error** depending on the
  system. Single runs ranged from 1.0× to 3.3×, so one run can't measure the
  size of the effect.
- **Mixing systems doesn't beat specialists yet.** With the same training
  budget, single-system models were more accurate on 7 of 8 systems in all
  5 seeds. Epidemics is a tie: 0.039 vs 0.037 on average, with the joint
  model ahead in 2 of 5 seeds. On ecology the joint model came close. On
  gene networks it was no better than the baseline.
- **Zero-shot is weak.** With no examples at all, it beat the baseline on
  neurons in all 5 seeds, on Kuramoto in 1 of 5, and on epidemics never.

At 1.08M parameters, this is the pattern you'd expect from a model that is
too small. The joint model has to fit 8 systems into the capacity each
specialist spends on one. The next experiment is to scale up the model and
the number of systems, and check whether the gap to the specialists closes.

## Scale-up, Stage A (2026-11-20)

The question: does a bigger model (1.08M → 8.5M → 28M parameters) close
the joint model's gap with the specialists? We trained 3 seeds per size on
ORCA, with success criteria fixed before running. Report:
[results_stageA/SCALING.md](results_stageA/SCALING.md).

- **No.** The gap went 1.45 → 1.54 → 1.52, so the pre-registered stop rule
  applies.
- **The joint model does get better with size** (24% lower error at 28M).
  The specialists just improve slightly faster (27%).
- **It's split by system.** The gap shrinks for Kuramoto, gene networks,
  neurons and epidemics. It grows for Lorenz96 and n-body, the continuous
  physics systems.
- **The joint models were still improving when training stopped**, more so
  when bigger. The leading explanation is that they're undertrained: with
  equal steps, each system gets 1/8 of a specialist's training. The next
  test is to train them 8× longer.

**Follow-up: 8× longer joint training (160k steps).** Report:
[results_stageA_joint160k/SCALING.md](results_stageA_joint160k/SCALING.md).

- **The joint model now beats the specialists at every size:** gap
  0.96 / 0.92 / 0.92, against 1.45 / 1.54 / 1.52 at 20k steps. Each system
  now gets as much training as a specialist does.
- **Lorenz96 and n-body**, the systems that got worse with size before,
  now improve with size. n-body ends up 30% better than its specialist.
- **28M beats 8.5M on validation likelihood**, though not on the CRPS gap.
- **Caveat:** the specialists (20k steps) and the joint models (160k
  steps) were both still improving when training stopped. This is a win at
  equal training per system, not at convergence. The next test is
  specialists trained 8× longer too.

**Correction (2026-11-21): with specialists also trained for 160k steps,
the specialists win.** Report:
[results_stageA_spec160k/CONVERGENCE.md](results_stageA_spec160k/CONVERGENCE.md).

- **The gap is 1.29 at 1.08M and 1.08 at 8.5M.** The joint model's win
  above came from comparing it against undertrained specialists.
- **The cost of sharing shrinks fast with size** (a 0.21 drop, against
  seed noise of about 0.01). At 8.5M it's close to a tie on 5 of 8
  systems, and most of the loss is n-body (1.36).
- **At 8.5M the specialists overfit (18 of 24) while the joint models are
  still improving.** The specialists are limited by data; the joint model
  isn't yet.
- **Open question:** does the gap close at 28M? The case for one model now
  rests on that size trend and on transfer to new systems.

**28M, and what "equal training" means (2026-11-21).**

- **At 28M the gap is 1.04:** the specialists still win, narrowly (the tie
  line is 1.03). The size benefit flattens: −0.21 from 1.08M to 8.5M, then
  −0.04 to 28M.
- **The joint model ties or wins on 6 of 8 systems** and beats the Lorenz96
  specialist (0.95). The whole remaining gap is **n-body (1.31) and
  epidemic (1.11)**, the systems whose specialists needed the most
  training.
- **Correction:** the 160k-step specialists did *not* get the same
  training per system as the joint model.
  - The joint model's 160k steps are shared by 8 systems (~20k each), so
    the 8 specialists used 8× the compute.
  - **At equal compute, the joint model wins** (0.92).
  - **Against specialists trained to their best, it's within 4% at 28M**,
    with 1/8 of the compute, and still improving.
- **Next test (proposed):** train the joint model 4× longer, to see whether
  n-body and epidemic catch up.

## Phase 1: real data (2026-11-21)

Does pretraining on the 8 simulators help forecast a real system? Four
datasets are planned and run one at a time: flu, neurons, baboon movement,
and power-grid frequency.

**US flu** (CDC ILINet, 10 HHS regions, weekly, 2002–2026). Report:
[results_phase1/flu_PHASE1.md](results_phase1/flu_PHASE1.md).

- **The model forecasts flu well:** 35–40% lower error than "same week last
  year", even from 3 seasons of data.
- **Pretraining on simulators didn't help** (fine-tuned / from-scratch
  0.95–1.07; the pre-registered bar was 0.90). This is the dataset with the
  closest simulator match.
- **The forecasts are overconfident:** the 90% range holds the truth only
  41–77% of the time.
- **Caveat:** fine-tuned models all peaked at the first validation check,
  so their best point was probably missed. A finer rerun is proposed.

## What's next (Phase 1)

- **Real data**, one field at a time:
  - ecology: Movebank animal tracks, eBird, NEON
  - neuroscience: Allen Brain Observatory, DANDI
  - cells: single-cell time courses
  - climate: ERA5
  - astronomy: Gaia
  - society: GDELT, World Bank
- **Variable sizes.** Lift the 12-entity / 4-feature cap with feature-name
  embeddings and sparse attention, so a system can have thousands of entities.
- **Scale metadata.** Tell the model the time step, units and spatial scale,
  so systems at very different scales can share one model.
- **Cross-scale links** (Phase 2). Summary tokens from a lower level (cells)
  become entities at the level above (organism, then population).
