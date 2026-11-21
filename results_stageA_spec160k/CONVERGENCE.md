# Joint vs specialists at equal training (160k steps each)

Joint models from `/home/pjallen5/lcm-runs/stageA_joint160k`; 160k-step specialists from `/home/pjallen5/lcm-runs/stageA_spec160k`; 20k-step specialists from `/home/pjallen5/lcm-runs/stageA_n20k`.
Gap = joint / specialist CRPS, geometric mean over 8 systems (below 1 = joint better).

## Verdict (criteria fixed before running)

Joint still wins: gap 0.97 or less. Tie: 0.97-1.03. Specialists win: above 1.03.

| Size | Seeds complete | Gap mean | sd | Per seed | Verdict | Gap vs 20k specialists |
|---|---|---|---|---|---|---|
| 1.08M | 3 of 3 | 1.289 | 0.021 | s0: 1.291, s1: 1.308, s2: 1.268 | **SPECIALISTS WIN** | 0.958 |
| 8.5M | 3 of 3 | 1.079 | 0.005 | s0: 1.079, s1: 1.084, s2: 1.074 | **SPECIALISTS WIN** | 0.917 |

## Gap per system (mean over seeds)

| System | 1.08M | 8.5M |
|---|---|---|
| predator_prey | 1.20 | 1.01 |
| boids | 1.14 | 1.03 |
| kuramoto | 1.32 | 1.10 |
| lorenz96 | 2.58 | 1.03 |
| nbody | 1.22 | 1.36 |
| gene_reg | 1.06 | 1.01 |
| neurons | 1.11 | 1.02 |
| epidemic | 1.12 | 1.11 |

## How much the specialists gained from 8x training

160k-step specialist CRPS / 20k-step specialist CRPS (below 1 = improved).

| System | 1.08M | 8.5M |
|---|---|---|
| predator_prey | 0.78 | 0.95 |
| boids | 0.83 | 0.91 |
| kuramoto | 0.78 | 0.91 |
| lorenz96 | 0.50 | 0.98 |
| nbody | 0.72 | 0.49 |
| gene_reg | 0.97 | 1.00 |
| neurons | 0.85 | 0.97 |
| epidemic | 0.63 | 0.75 |

## Specialist validation curves

| Size | Overfitting* | Still improving at the end | Runs |
|---|---|---|---|
| 1.08M | 3 | 15 | 24 |
| 8.5M | 18 | 3 | 24 |

*Validation NLL rose more than 0.05 after its best point; the best checkpoint was used anyway.
