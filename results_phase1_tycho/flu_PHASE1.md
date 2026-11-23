# Phase 1: flu

Does pretraining on 8 simulated systems help forecast this real system? CRPS in normalised units, mean ± sd over seeds; lower is better.

## Small-data regime (verdict)

Windows: {'train': 93, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | Pretraining | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds | No fine-tuning | Coverage (scratch) | Best step scratch / fine-tuned |
|---|---|---|---|---|---|---|---|---|---|
| 8.5M | sims | 0.294 ± 0.004 | 0.281 ± 0.002 | 0.954 | 0.70 | scratch 3, finetune 3 | 0.492 | 0.62 | 450,550,425 / 75,75,75 |
| 8.5M | simtycho | 0.294 ± 0.004 | 0.270 ± 0.004 | 0.918 | 0.70 | scratch 3, finetune 3 | 0.362 | 0.62 | 450,550,425 / 50,50,50 |
| 8.5M | tycho | 0.294 ± 0.004 | 0.245 ± 0.005 | 0.833 | 0.73 | scratch 3, finetune 3 | 0.319 | 0.62 | 450,550,425 / 100,100,100 |

90% coverage by season of the forecast start (mean over windows and seeds; target 0.90):

| Season | 8.5M finetune-simtycho | 8.5M finetune-tycho | 8.5M finetune | 8.5M scratch |
|---|---|---|---|---|
| 2015-16 | 0.81 | 0.76 | 0.83 | 0.81 |
| 2016-17 | 0.77 | 0.81 | 0.80 | 0.82 |
| 2017-18 | 0.74 | 0.78 | 0.65 | 0.72 |
| 2018-19 | 0.76 | 0.79 | 0.69 | 0.59 |
| 2019-20 | 0.58 | 0.60 | 0.57 | 0.53 |
| 2020-21 | 0.72 | 0.71 | 0.78 | 0.72 |
| 2021-22 | 0.59 | 0.47 | 0.62 | 0.46 |
| 2022-23 | 0.68 | 0.76 | 0.70 | 0.55 |
| 2023-24 | 0.78 | 0.83 | 0.81 | 0.64 |
| 2024-25 | 0.67 | 0.80 | 0.68 | 0.54 |
| 2025-26 | 0.69 | 0.78 | 0.66 | 0.56 |

### Verdict (criteria fixed before running)

- **8.5M (sims): FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.281 vs 0.294 (ratio 0.954))
  - Beats persistence: yes (0.281 vs 0.593)
  - Beats seasonal-naive: yes (0.281 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.70)
- **8.5M (simtycho): FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.270 vs 0.294 (ratio 0.918))
  - Beats persistence: yes (0.270 vs 0.593)
  - Beats seasonal-naive: yes (0.270 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.70)
  - Beats the simulator-pretrained fine-tuned model: yes (0.270 vs 0.281)
- **8.5M (tycho): FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): yes (0.245 vs 0.294 (ratio 0.833))
  - Beats persistence: yes (0.245 vs 0.593)
  - Beats seasonal-naive: yes (0.245 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.73)
  - Beats the simulator-pretrained fine-tuned model: yes (0.245 vs 0.281)

## Full-data regime

Windows: {'train': 511, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | Pretraining | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds | No fine-tuning | Coverage (scratch) | Best step scratch / fine-tuned |
|---|---|---|---|---|---|---|---|---|---|
| 8.5M | sims | 0.266 ± 0.010 | 0.256 ± 0.010 | 0.961 | 0.76 | scratch 3, finetune 3 | 0.492 | 0.71 | 925,900,950 / 100,100,125 |
| 8.5M | simtycho | 0.266 ± 0.010 | 0.255 ± 0.014 | 0.957 | 0.70 | scratch 3, finetune 3 | 0.362 | 0.71 | 925,900,950 / 100,100,125 |
| 8.5M | tycho | 0.266 ± 0.010 | 0.240 ± 0.014 | 0.900 | 0.78 | scratch 3, finetune 3 | 0.319 | 0.71 | 925,900,950 / 450,425,250 |

90% coverage by season of the forecast start (mean over windows and seeds; target 0.90):

| Season | 8.5M finetune-simtycho | 8.5M finetune-tycho | 8.5M finetune | 8.5M scratch |
|---|---|---|---|---|
| 2015-16 | 0.82 | 0.84 | 0.87 | 0.84 |
| 2016-17 | 0.83 | 0.88 | 0.86 | 0.90 |
| 2017-18 | 0.75 | 0.82 | 0.79 | 0.80 |
| 2018-19 | 0.76 | 0.86 | 0.80 | 0.72 |
| 2019-20 | 0.60 | 0.66 | 0.69 | 0.50 |
| 2020-21 | 0.75 | 0.72 | 0.81 | 0.68 |
| 2021-22 | 0.59 | 0.62 | 0.62 | 0.63 |
| 2022-23 | 0.72 | 0.86 | 0.76 | 0.77 |
| 2023-24 | 0.73 | 0.80 | 0.80 | 0.79 |
| 2024-25 | 0.65 | 0.84 | 0.71 | 0.65 |
| 2025-26 | 0.66 | 0.75 | 0.69 | 0.59 |
