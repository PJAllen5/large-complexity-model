# Real datasets (Phase 1)

## flu_hhs_wili.csv

- **What:** CDC ILINet weighted influenza-like illness (wILI), the percentage of
  outpatient doctor visits for flu-like illness, weighted by state population.
  Weekly, for the 10 US HHS regions.
- **Range:** epiweek 2002w40 to 2026w45 (1,259 weeks, no gaps). The data before
  2002 has no reports in summer weeks, so it was left out.
- **Source:** Delphi Epidata API, FluView endpoint
  (https://cmu-delphi.github.io/delphi-epidata/api/fluview.html), fetched
  2026-11-21:
  `https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,...,hhs10&epiweeks=200240-202652`
- **Licence:** US government data (publicly accessible). The values are the latest
  revised figures at fetch time.
- **Loader:** `lcm/real.py`, `load_flu()`.
