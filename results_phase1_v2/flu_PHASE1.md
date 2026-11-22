# Phase 1: flu

Does pretraining on 8 simulated systems help forecast this real system? CRPS in normalised units, mean ± sd over seeds; lower is better.

## Small-data regime (verdict)

Windows: {'train': 93, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds | No fine-tuning | Coverage (scratch) | Best step scratch / fine-tuned |
|---|---|---|---|---|---|---|---|---|
| 28M | 0.292 ± 0.014 | 0.309 ± 0.008 | 1.060 | 0.63 | scratch 3, finetune 3 | 0.498 | 0.59 | 375,550,400 / 75,50,75 |
| 8.5M | 0.294 ± 0.004 | 0.281 ± 0.002 | 0.954 | 0.70 | scratch 3, finetune 3 | 0.492 | 0.62 | 450,550,425 / 75,75,75 |

90% coverage by season of the forecast start (mean over windows and seeds; target 0.90):

| Season | 28M finetune | 28M scratch | 8.5M finetune | 8.5M scratch |
|---|---|---|---|---|
| 2015-16 | 0.81 | 0.77 | 0.83 | 0.81 |
| 2016-17 | 0.73 | 0.77 | 0.80 | 0.82 |
| 2017-18 | 0.58 | 0.74 | 0.65 | 0.72 |
| 2018-19 | 0.60 | 0.62 | 0.69 | 0.59 |
| 2019-20 | 0.51 | 0.51 | 0.57 | 0.53 |
| 2020-21 | 0.73 | 0.66 | 0.78 | 0.72 |
| 2021-22 | 0.60 | 0.40 | 0.62 | 0.46 |
| 2022-23 | 0.65 | 0.51 | 0.70 | 0.55 |
| 2023-24 | 0.72 | 0.62 | 0.81 | 0.64 |
| 2024-25 | 0.59 | 0.50 | 0.68 | 0.54 |
| 2025-26 | 0.60 | 0.53 | 0.66 | 0.56 |

### Verdict (criteria fixed before running)

- **28M: FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.309 vs 0.292 (ratio 1.060))
  - Beats persistence: yes (0.309 vs 0.593)
  - Beats seasonal-naive: yes (0.309 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.63)
- **8.5M: FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.281 vs 0.294 (ratio 0.954))
  - Beats persistence: yes (0.281 vs 0.593)
  - Beats seasonal-naive: yes (0.281 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.70)

## Full-data regime

Windows: {'train': 511, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds | No fine-tuning | Coverage (scratch) | Best step scratch / fine-tuned |
|---|---|---|---|---|---|---|---|---|
| 28M | 0.272 ± 0.016 | 0.260 ± 0.005 | 0.958 | 0.72 | scratch 3, finetune 3 | 0.498 | 0.69 | 850,750,875 / 125,100,125 |
| 8.5M | 0.266 ± 0.010 | 0.256 ± 0.010 | 0.961 | 0.76 | scratch 3, finetune 3 | 0.492 | 0.71 | 925,900,950 / 100,100,125 |

90% coverage by season of the forecast start (mean over windows and seeds; target 0.90):

| Season | 28M finetune | 28M scratch | 8.5M finetune | 8.5M scratch |
|---|---|---|---|---|
| 2015-16 | 0.86 | 0.83 | 0.87 | 0.84 |
| 2016-17 | 0.85 | 0.88 | 0.86 | 0.90 |
| 2017-18 | 0.73 | 0.79 | 0.79 | 0.80 |
| 2018-19 | 0.80 | 0.73 | 0.80 | 0.72 |
| 2019-20 | 0.63 | 0.51 | 0.69 | 0.50 |
| 2020-21 | 0.78 | 0.68 | 0.81 | 0.68 |
| 2021-22 | 0.60 | 0.62 | 0.62 | 0.63 |
| 2022-23 | 0.72 | 0.75 | 0.76 | 0.77 |
| 2023-24 | 0.78 | 0.76 | 0.80 | 0.79 |
| 2024-25 | 0.67 | 0.63 | 0.71 | 0.65 |
| 2025-26 | 0.65 | 0.56 | 0.69 | 0.59 |
