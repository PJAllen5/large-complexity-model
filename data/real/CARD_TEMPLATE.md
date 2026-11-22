# Data card: <dataset name>

Fill this in **before any training** on a new real dataset, and commit it with
the loader. Each answer is short; "unknown" is allowed but must say what would
settle it.

## 1. Source and licence
- **What it measures** (and what it's a proxy for):
- **Source and exact query or files** (URL, version, fetch date):
- **Licence / terms:**
- **Can the data go in this public repo?** (yes / no / only derived summaries) and why:
- **Attribution required?**

## 2. Sensitivity
- **People:** is any of it about individuals, or could it identify them?
- **Animals or sites:** could it expose the location of vulnerable species or places?
- **Decision:** what is kept out (or aggregated) because of this:

## 3. Leakage audit
- **Split** (by time, entity or recording) and exact boundaries:
- **Overlap with any pretraining data** (same source, same period, same entities, aggregated versions of the same data):
- **Correlated outside data** that could carry test-period information:
- **Cutoff applied to pretraining data:**

## 4. Format choices (fitting into 12 entities x 4 features, regular steps)
- **Entities** (which, how chosen, how many available):
- **Features:**
- **Time step, and how irregular sampling is handled:**
- **Missing data** (how much; windows dropped; could dropping bias toward calm periods?):
- **Interaction graph** (how defined):

## 5. Measurement quirks
- Known artefacts (reporting changes, holidays, sensor drift, revisions):
- How a model could learn the instrument rather than the system:

## 6. Baselines
- Field-appropriate simple baselines (and why):
- Any published or off-the-shelf forecasting model to compare against:

## 7. Size and storage
- Raw size; size after conversion; where it lives (repo, ~/lcm-runs, /scratch):
