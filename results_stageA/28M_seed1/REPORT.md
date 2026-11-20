# LCM Phase 0 results

Config: 20000 training episodes per system, 20000 training steps per model, 128 test episodes per system. Numbers are in per-episode normalised units.

## The three questions

1. **Does one model trained on all systems beat specialists?** The joint model beats the specialist on **0 of 8** systems (none), with the same training budget.
2. **Can it forecast a system it has never seen?** Zero-shot it beats the 'nothing changes' baseline on **1 of 3** held-out systems (neurons).
3. **Does prior experience make new systems faster to learn?** With only 256 examples, the pretrained LCM beats a fresh model on **3 of 3** (neurons, epidemic, kuramoto).

Odds: across situations, the joint model's odds are off by **0.13** on average, against 0.40 for an informed guess that always says the system's average odds. The truth lands inside its 90% range **87%** of the time (ideal: 90%).

![skill](fig_skill.png)

![transfer](fig_transfer.png)

![odds](fig_odds.png)

![forecasts](fig_forecasts.png)

## Forecast accuracy per system

CRPS scores the whole predicted distribution (lower is better). The baseline repeats the last observation.

| system | field | baseline | specialist | joint LCM | LCM, never saw it | joint 90% coverage |
|---|---|---|---|---|---|---|
| predator_prey | ecology | 0.091 | 0.037 | 0.046 | - | 86% |
| boids | animal behaviour | 0.484 | 0.139 | 0.176 | - | 84% |
| kuramoto | synchronisation | 0.540 | 0.097 | 0.137 | 0.686 | 87% |
| lorenz96 | climate / chaos | 0.763 | 0.044 | 0.183 | - | 88% |
| nbody | astronomy | 0.584 | 0.081 | 0.188 | - | 89% |
| gene_reg | cell biology | 0.073 | 0.049 | 0.051 | - | 87% |
| neurons | neuroscience | 1.097 | 0.070 | 0.083 | 0.392 | 85% |
| epidemic | society / epidemiology | 0.231 | 0.010 | 0.011 | 0.361 | 87% |

## Learning a new system from little data

| held-out system | fresh model, few examples | LCM, zero examples | LCM + few examples | specialist, full data |
|---|---|---|---|---|
| neurons | 0.100 | 0.392 | 0.089 | 0.070 |
| epidemic | 0.034 | 0.361 | 0.018 | 0.010 |
| kuramoto | 0.159 | 0.686 | 0.151 | 0.097 |

## Odds of an outcome

Event: *the key variable of entity 0 ends the forecast window higher than it is now.* True odds come from replaying each situation 64 times in the simulator with fresh randomness. The informed guess knows each system's average odds; the LCM has to work out the odds for each situation.

| system | LCM odds error | informed-guess error |
|---|---|---|
| predator_prey | 0.206 | 0.346 |
| boids | 0.084 | 0.474 |
| kuramoto | 0.095 | 0.436 |
| lorenz96 | 0.194 | 0.481 |
| nbody | 0.236 | 0.437 |
| gene_reg | 0.120 | 0.224 |
| neurons | 0.048 | 0.404 |
| epidemic | 0.054 | 0.367 |
