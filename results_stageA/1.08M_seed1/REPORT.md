# LCM Phase 0 results

Config: 20000 training episodes per system, 20000 training steps per model, 128 test episodes per system. Numbers are in per-episode normalised units.

## The three questions

1. **Does one model trained on all systems beat specialists?** The joint model beats the specialist on **0 of 8** systems (none), with the same training budget.
2. **Can it forecast a system it has never seen?** Zero-shot it beats the 'nothing changes' baseline on **1 of 3** held-out systems (neurons).
3. **Does prior experience make new systems faster to learn?** With only 256 examples, the pretrained LCM beats a fresh model on **3 of 3** (neurons, epidemic, kuramoto).

Odds: across situations, the joint model's odds are off by **0.18** on average, against 0.40 for an informed guess that always says the system's average odds. The truth lands inside its 90% range **88%** of the time (ideal: 90%).

![skill](fig_skill.png)

![transfer](fig_transfer.png)

![odds](fig_odds.png)

![forecasts](fig_forecasts.png)

## Forecast accuracy per system

CRPS scores the whole predicted distribution (lower is better). The baseline repeats the last observation.

| system | field | baseline | specialist | joint LCM | LCM, never saw it | joint 90% coverage |
|---|---|---|---|---|---|---|
| predator_prey | ecology | 0.091 | 0.044 | 0.053 | - | 85% |
| boids | animal behaviour | 0.484 | 0.165 | 0.202 | - | 85% |
| kuramoto | synchronisation | 0.540 | 0.122 | 0.186 | 0.664 | 88% |
| lorenz96 | climate / chaos | 0.763 | 0.086 | 0.301 | - | 88% |
| nbody | astronomy | 0.584 | 0.186 | 0.262 | - | 92% |
| gene_reg | cell biology | 0.073 | 0.049 | 0.056 | - | 92% |
| neurons | neuroscience | 1.097 | 0.080 | 0.098 | 0.322 | 88% |
| epidemic | society / epidemiology | 0.231 | 0.015 | 0.017 | 0.514 | 89% |

## Learning a new system from little data

| held-out system | fresh model, few examples | LCM, zero examples | LCM + few examples | specialist, full data |
|---|---|---|---|---|
| neurons | 0.154 | 0.322 | 0.100 | 0.080 |
| epidemic | 0.061 | 0.514 | 0.049 | 0.015 |
| kuramoto | 0.183 | 0.664 | 0.168 | 0.122 |

## Odds of an outcome

Event: *the key variable of entity 0 ends the forecast window higher than it is now.* True odds come from replaying each situation 64 times in the simulator with fresh randomness. The informed guess knows each system's average odds; the LCM has to work out the odds for each situation.

| system | LCM odds error | informed-guess error |
|---|---|---|
| predator_prey | 0.234 | 0.346 |
| boids | 0.098 | 0.474 |
| kuramoto | 0.140 | 0.436 |
| lorenz96 | 0.268 | 0.481 |
| nbody | 0.335 | 0.437 |
| gene_reg | 0.180 | 0.224 |
| neurons | 0.093 | 0.404 |
| epidemic | 0.068 | 0.367 |
