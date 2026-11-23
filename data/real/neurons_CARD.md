# Data card: Allen Brain Observatory neurons (mouse visual cortex, 2-photon ΔF/F)

Written 2026-11-22, before any training. Phase 1 dataset 2 (after flu).

## 1. Source and licence
- **What it measures:** ΔF/F, the change in a neuron's calcium-indicator (GCaMP6) fluorescence relative to its baseline. A slow, smoothed proxy for its spiking activity, recorded at about 30 frames per second in awake mice watching visual stimuli.
- **Source:** Allen Brain Observatory – Visual Coding, 2-photon (Allen Institute). Public AWS bucket `s3://allen-brain-observatory` (https://registry.opendata.aws/allen-brain-observatory/): the manifest `visual-coding-2p/ophys_experiments.json` and one NWB file per recording, `visual-coding-2p/ophys_experiment_data/<id>.nwb`. Fetched by `python -m lcm.neurons fetch`; the chosen recording ids are saved in `neurons_summary.json`.
- **Licence / terms:** Allen Institute Terms of Use (https://alleninstitute.org/terms-of-use/, checked 2026-11-22; page dated 23 October 2026). Use, copying, distribution and derivative works are allowed "for research or other noncommercial purposes". Commercial use or redistribution needs written permission.
- **Can the data go in this public repo?** No. Derived non-commercial redistribution looks allowed, but the terms don't address it directly, so nothing is committed: the repo has the fetch code; the data lives in `~/lcm-runs/neurons` on ORCA.
- **Attribution required:** yes, per the Allen Citation Policy (https://alleninstitute.org/citation-policy/). Cite the primary publication and the dataset wherever results appear:
  - de Vries SEJ, Lecoq JA, Buice MA, et al. A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23, 138–151 (2020).
  - "Allen Brain Observatory – Visual Coding AWS Public Data Set was accessed on 2026-11-22 from https://registry.opendata.aws/allen-brain-observatory."

## 2. Sensitivity
- **People:** none. Mice; no personal data.
- **Animals or sites:** no location or welfare concern from using published recordings.
- **Decision:** nothing kept out for sensitivity.

## 3. Leakage audit
- **Split: by mouse.** Session type A recordings, one per mouse, in a fixed shuffled order (seed 0); recordings with fewer than 12 neurons are skipped. The first 30 usable go to training, the next 5 to validation, the next 10 to test. No mouse is in two splits.
- **Overlap with pretraining data:** none. The simulators are synthetic (they include FitzHugh–Nagumo model neurons, a different and much simpler kind of signal); Tycho is US disease counts.
- **Correlated outside data:** every recording sees the same stimulus sequence type (session A), so test mice see stimuli of the same kind as training mice. That's intended: the question is forecasting new animals, not new stimuli.
- **Cutoff:** not applicable (no shared timeline).

## 4. Format choices
- **Entities:** 12 neurons per recording, picked at random with a seed equal to the recording id (not chosen by activity, which would favour easy, busy cells). Recordings vary widely in neuron count; the inspected sample (experiment 593506468, the smallest file) had 1.
- **Features:** 1 (ΔF/F).
- **Time step:** 4 frames averaged, about 7.5 Hz (0.133 s). Window 64 steps = 8.5 s: 6.4 s of history, 2.1 s to forecast. The inspected file had no gaps (every frame 0.033 s apart) and no missing values.
- **Missing data:** windows with any non-finite value are dropped (`to_episodes`). None expected.
- **Windows:** training every 16 steps (about 1,900 per recording); validation and test every 256 steps (about 120 per recording, about 600 validation and 1,200 test). Small-data regime: 25 evenly spaced windows from each of the first 4 training recordings (100 windows; flu's was 93).
- **Interaction graph:** every pair of the 12 neurons linked. The real connections between imaged neurons aren't known.

## 5. Measurement quirks
- ΔF/F is spiky: in the inspected file the median was 0.01 and the top 1% above 2.9. Most of the time neurons sit near zero, then flash. Calcium decays over about 0.5–1 s, slower than spikes.
- A session mixes stimuli (in session A: drifting gratings, natural movies, spontaneous activity), and each changes activity. Windows that span a stimulus change are kept.
- Neuropil contamination and motion are corrected by Allen's pipeline; residual artefacts can remain.
- **Instrument risk:** the model may learn the indicator's rise-and-decay shape (a property of GCaMP6) more than the neurons' dynamics.

## 6. Baselines
- **Persistence** (the last value).
- **History mean** (each neuron's mean over its 48-step history), in place of seasonal-naive, since there is no period. Decided before any data was used: it's a fairer simple rival than persistence for spiky activity.
- No published forecasting model on this exact setup to compare against.

## 7. Size and storage
- NWB files about 100–800 MB each (median about 280 MB); 1,518 in total, 506 of session type A. About 45–60 downloaded (some are skipped), each deleted right after its 12 traces are kept.
- Kept: about 1.5 MB per recording before compression, about 70 MB in total, in `~/lcm-runs/neurons` on ORCA. Not in the repo.
