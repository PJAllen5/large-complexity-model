# Phase 1: flu

Does pretraining on 8 simulated systems help forecast this real system? CRPS in normalised units, mean ± sd over seeds; lower is better.

## Small-data regime (verdict)

Windows: {'train': 93, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds |
|---|---|---|---|---|---|
| 28M | 0.309 ± 0.009 | 0.320 ± 0.021 | 1.035 | 0.46 | scratch 3, finetune 3 |
| 8.5M | 0.303 ± 0.004 | 0.288 ± 0.019 | 0.952 | 0.53 | scratch 3, finetune 3 |

### Verdict (criteria fixed before running)

- **28M: FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.320 vs 0.309 (ratio 1.035))
  - Beats persistence: yes (0.320 vs 0.593)
  - Beats seasonal-naive: yes (0.320 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.46)
- **8.5M: FAIL**
  - Pretraining helps (fine-tuned <= 0.90 x scratch): NO (0.288 vs 0.303 (ratio 0.952))
  - Beats persistence: yes (0.288 vs 0.593)
  - Beats seasonal-naive: yes (0.288 vs 0.469)
  - Calibrated (90% range holds truth 80-95%): NO (0.53)

## Full-data regime

Windows: {'train': 511, 'val': 42, 'test': 517}. Baselines (CRPS): Persistence 0.593; seasonal-naive 0.469.

| Size | From scratch | Fine-tuned | Fine-tuned / scratch | Coverage (fine-tuned) | Seeds |
|---|---|---|---|---|---|
| 28M | 0.288 ± 0.037 | 0.309 ± 0.010 | 1.071 | 0.58 | scratch 3, finetune 3 |
| 8.5M | 0.284 ± 0.006 | 0.291 ± 0.014 | 1.024 | 0.61 | scratch 3, finetune 3 |
