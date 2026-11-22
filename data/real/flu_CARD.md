# Data card: US flu (CDC ILINet wILI, 10 HHS regions)

Written 2026-11-22, after the first two flu runs (the template didn't exist yet).

## 1. Source and licence
- **What it measures:** weighted % of outpatient doctor visits for influenza-like illness (fever plus cough or sore throat), weekly. It's a **proxy** for flu activity: it also catches other respiratory viruses (including COVID from 2020) and depends on who visits doctors.
- **Source:** Delphi Epidata API, FluView endpoint; query in [README.md](README.md); fetched 2026-11-21.
- **Licence:** US government data, publicly accessible (Delphi lists it as such).
- **In this public repo?** Yes: `flu_hhs_wili.csv`, 110 KB.
- **Attribution:** cite CDC ILINet and the Delphi Epidata API.

## 2. Sensitivity
- **People:** aggregated to 10 multi-state regions; no individual data.
- **Animals or sites:** not applicable.
- **Decision:** nothing kept out.

## 3. Leakage audit
- **Split by flu season (week 40 to week 39):** train 2002-03 to 2012-13 (small: 2010-11 to 2012-13), validation 2013-14 and 2014-15, test 2015-16 onward. Windows lie entirely inside one split.
- **Overlap with pretraining data:** none for the simulator pretraining.
- **Correlated outside data:** US state-level ILINet (the same data split finer), flu in other countries (same strains; Southern Hemisphere season first), COVID data (from 2020). **Any real-data pretraining must end before 2015w40.**
- **Cutoff applied:** not needed so far (simulators only).

## 4. Format choices
- **Entities:** all 10 HHS regions (no selection).
- **Features:** 1 (wILI). Patient counts, age groups and unweighted ILI are available but unused.
- **Time step:** 1 week, regular.
- **Missing data:** none from 2002w40; earlier seasons have no summer reports, so 1997-2002 was left out.
- **Graph:** regions that share a land border (16 pairs).

## 5. Measurement quirks
- Holiday dips around Christmas; provider network growth over the years; definition and reporting changes; COVID-era behaviour changes (fewer visits, then COVID itself counted as ILI); values are revised after first release (we use the latest).
- **Instrument risk:** the model could learn holiday and reporting patterns rather than transmission; the post-2020 shifts are the largest such change and are where coverage is worst.

## 6. Baselines
- Persistence (last week) and seasonal-naive (same week last year).
- Not yet compared: published CDC FluSight forecasts, or a general time-series model such as Chronos.

## 7. Size and storage
- 1,259 weeks x 10 regions; 110 KB CSV in the repo.
