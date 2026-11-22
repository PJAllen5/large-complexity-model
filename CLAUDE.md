# LCM: notes for Claude

## Project log (required)

Record every decision, test, experiment run and result in
[PROJECT_LOG.md](PROJECT_LOG.md), in the same change as the work itself:

- Add a dated entry at the bottom of the **Timeline** (`### YYYY-MM-DD: title`).
- Say **what** was decided or run, **why**, and **what came out of it**,
  including failures, bugs fixed and options the user declined.
- For experiment runs, include the config, hardware, run time, cost (if
  rented) and headline numbers, and link the report.
- If model size or settings change, update the **Reference** tables at the top.
- Correct earlier entries by adding a new entry that says what was wrong.
  Don't silently rewrite history.

## Project basics

- Python environment: `.venv/` (not committed). To recreate it:
  `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Quick check: `.venv/bin/python experiment.py --quick` (~7 min on an M3).
- Full run: `.venv/bin/python experiment.py` (~82 min on an M3). Finished
  runs are cached in `results/runs/`, so a stopped run resumes.
- GitHub repo is private: PJAllen5/large-complexity-model.

## Real datasets (Phase 1)

- Before training on a new real dataset, fill in a data card from
  `data/real/CARD_TEMPLATE.md` and commit it with the loader. Check the
  licence before committing any data: this repo is public.
