# Joint vs specialists at equal training (160k steps each)

Joint models from `/home/pjallen5/lcm-runs/stageA_joint160k`; 160k-step specialists from `/home/pjallen5/lcm-runs/stageA_spec160k`; 20k-step specialists from `/home/pjallen5/lcm-runs/stageA_n20k`.
Gap = joint / specialist CRPS, geometric mean over 8 systems (below 1 = joint better).

## Verdict (criteria fixed before running)

Joint still wins: gap 0.97 or less. Tie: 0.97-1.03. Specialists win: above 1.03.

| Size | Seeds complete | Gap mean | sd | Per seed | Verdict | Gap vs 20k specialists |
|---|---|---|---|---|---|---|
| 1.08M | 3 of 3 | 1.289 | 0.021 | s0: 1.291, s1: 1.308, s2: 1.268 | **SPECIALISTS WIN** | 0.958 |
| 8.5M | 3 of 3 | 1.079 | 0.005 | s0: 1.079, s1: 1.084, s2: 1.074 | **SPECIALISTS WIN** | 0.917 |
| 28M | 3 of 3 | 1.042 | 0.017 | s0: 1.028, s1: 1.062, s2: 1.038 | **SPECIALISTS WIN** | 0.919 |

## Gap per system (mean over seeds)

| System | 1.08M | 8.5M | 28M |
|---|---|---|---|
| predator_prey | 1.20 | 1.01 | 1.01 |
| boids | 1.14 | 1.03 | 1.00 |
| kuramoto | 1.32 | 1.10 | 0.99 |
| lorenz96 | 2.58 | 1.03 | 0.95 |
| nbody | 1.22 | 1.36 | 1.31 |
| gene_reg | 1.06 | 1.01 | 0.99 |
| neurons | 1.11 | 1.02 | 1.01 |
| epidemic | 1.12 | 1.11 | 1.11 |

## How much the specialists gained from 8x training

160k-step specialist CRPS / 20k-step specialist CRPS (below 1 = improved).

| System | 1.08M | 8.5M | 28M |
|---|---|---|---|
| predator_prey | 0.78 | 0.95 | 0.95 |
| boids | 0.83 | 0.91 | 0.96 |
| kuramoto | 0.78 | 0.91 | 0.98 |
| lorenz96 | 0.50 | 0.98 | 1.01 |
| nbody | 0.72 | 0.49 | 0.54 |
| gene_reg | 0.97 | 1.00 | 1.01 |
| neurons | 0.85 | 0.97 | 0.97 |
| epidemic | 0.63 | 0.75 | 0.78 |

## Specialist validation curves

| Size | Overfitting* | Still improving at the end | Runs |
|---|---|---|---|
| 1.08M | 3 | 15 | 24 |
| 8.5M | 18 | 3 | 24 |
| 28M | 20 | 4 | 24 |

*Validation NLL rose more than 0.05 after its best point; the best checkpoint was used anyway.
