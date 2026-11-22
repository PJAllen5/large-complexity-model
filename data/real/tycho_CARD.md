# Data card: Project Tycho (US weekly cases of other diseases, by state)

Written 2026-11-22, before any training. Used for **pretraining only** (the
Phase 1 test of real-data pretraining on flu).

## 1. Source and licence
- **What it measures:** weekly reported cases (and incidence per 100,000, from historical US Census populations) of nationally notifiable diseases, by state. Reported cases are a proxy for infections: reporting completeness varies by state and decade.
- **Source:** Project Tycho Level 1 v1.0.0, University of Pittsburgh. Zenodo record 12608992 (https://zenodo.org/records/12608992), file `ProjectTycho_Level1_v1.0.0.zip`, MD5 `4c3991594d8dbad117e69ef49808eb5b`. Fetched 2026-11-22 by `python -m lcm.tycho fetch`.
- **Licence:** Creative Commons Attribution (the readme's licence text is CC BY; Zenodo lists CC BY 4.0). Any use, with Project Tycho cited as the origin, the licence linked and changes marked. The readme also *requests* (not requires) that derived products be shared with them within three months.
- **Citation:** van Panhuis WG, Grefenstette J, Jung SY, Chok NS, Cross A, Eng H, Lee BY, Zadorozhny V, Brown S, Cummings D, Burke DS. Contagious Diseases in the United States from 1888 to the present. NEJM 2013; 369(22): 2152-2158.
- **Changes we make:** state-level rows only; selection of states and weeks (see 4); cut into 64-week windows.
- **In this public repo?** No (30.9 MB CSV). The repo has the fetch/build code; the data lives in `~/lcm-runs/tycho`.

## 2. Sensitivity
- **People:** state-level weekly counts only; no individual data.
- **Decision:** nothing kept out for sensitivity.

## 3. Leakage audit
- **Pretraining-internal split:** validation = two whole census divisions (Mid-Atlantic: NJ, NY, PA; Pacific: AK, CA, HI, OR, WA). Same years as training, so national epidemics are shared; validation only picks the pretraining checkpoint, and is somewhat optimistic.
- **Against the flu test (2015w40 onward):** all Tycho data ends by 2011 (windows used: 1928-2010). Different diseases. No overlap with ILINet.
- **Correlated outside data:** none relevant to 2015+ US flu.
- **Cutoff applied:** none needed (everything predates 2015w40).

## 4. Format choices
- **Entities:** one US census division per episode (3-9 states); only states that reported **every** week of the window (at least 3). Missing states are masked out. Average about 4-5 states per window.
- **Features:** 1 (incidence per 100,000).
- **Time step:** MMWR epi week (week 1 contains 4 January; 53-week years handled; checked against the flu file).
- **Missing data:** a week with no row is unreported (zero-case weeks are explicit rows). 8-43% of weeks are missing per state, by disease.
- **Selection:** windows with fewer than 25% of (state, week) values above zero are dropped (near-empty periods). This, and requiring complete reporting, favours well-reporting states and active epidemic periods.
- **Diseases kept** (windows train / val): smallpox 2,445 / 471; polio 1,834 / 573; pertussis 1,421 / 427; measles 1,261 / 488; mumps 247 / 52; hepatitis A 195 / 70. **Rubella left out** (14 windows). Diphtheria is city-level only and not used.
- **Graph:** every pair of the window's states linked (fully connected within a division). Flu uses shared land borders, so the graph meaning differs.

## 5. Measurement quirks
- Reporting changes across a century; under-reporting; vaccine introductions (polio 1955, measles 1963, mumps 1967) change dynamics sharply; smallpox disappears in the late 1940s.
- **Instrument risk:** windows cluster in a few eras per disease (e.g. polio 1928-1938), so the model may learn era-specific reporting rather than general epidemic dynamics.

## 6. Baselines
- Not evaluated on Tycho itself; its use is pretraining. The test is fine-tuned flu performance against from-scratch and simulator-pretrained models.

## 7. Size and storage
- Download 3.6 MB zip; CSV 30.9 MB; built windows about 1.1 MB (compressed npz). In `~/lcm-runs/tycho` on ORCA; not in the repo.
