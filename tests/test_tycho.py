"""Checks for the Project Tycho builder (lcm/tycho.py) on synthetic data."""

import csv
import io
import json
import os
import zipfile

import numpy as np
import pytest

from lcm import tycho
from lcm.data import T_TOTAL


def test_week_index_matches_the_flu_calendar():
    weeks = [int(r["epiweek"]) for r in csv.DictReader(open(os.path.join("data", "real", "flu_hhs_wili.csv")))]
    assert {tycho.week_index(b) - tycho.week_index(a) for a, b in zip(weeks, weeks[1:])} == {1}
    assert tycho.week_index(200853) + 1 == tycho.week_index(200901)        # a 53-week year
    assert tycho.index_year(tycho.week_index(195010)) == 1950


def _series(n_weeks=120, states=("A", "B", "C", "D"), gap=None, value=1.0):
    s = {st: {w: value for w in range(n_weeks)} for st in states}
    if gap:
        st, w = gap
        del s[st][w]
    return s


def test_windows_keep_only_fully_reported_states():
    wins = tycho.windows(_series(gap=("D", 70)), ["A", "B", "C", "D"])
    by_start = {w[0]: w[1] for w in wins}
    assert by_start[0] == ["A", "B", "C", "D"]                  # weeks 0-63: D complete
    assert by_start[10] == ["A", "B", "C"]                      # weeks 10-73 include D's gap at 70
    assert len(wins) == 120 - T_TOTAL + 1 and wins[0][2].shape == (T_TOTAL, 4)


def test_windows_need_three_states_and_some_activity():
    two = _series(gap=("C", 5))
    del two["D"]
    assert all(w[0] > 5 for w in tycho.windows(two, ["A", "B", "C"]))   # with C's gap only A and B are full
    assert tycho.windows(_series(value=0.0), ["A", "B", "C", "D"]) == []  # no cases at all
    assert tycho.windows(_series(), ["A", "B", "X"]) == []               # only 2 of the states exist


def test_universal_format_masks_and_links():
    wins = [(0, ["A", "B", "C"], np.ones((T_TOTAL, 3), np.float32))]
    d = tycho.to_universal(wins)
    assert d["x"].shape == (1, T_TOTAL, 12, 4) and d["ent_mask"][0].tolist() == [1, 1, 1] + [0] * 9
    assert d["adj"][0, :3, :3].tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]] and d["adj"][0, 3:].sum() == 0
    assert d["feat_mask"][0].tolist() == [1, 0, 0, 0]


def _write_csv(folder, rows):
    with open(os.path.join(folder, tycho.CSV_NAME), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epi_week", "state", "loc", "loc_type", "disease", "cases", "incidence_per_100000"])
        w.writerows(rows)


def test_build_splits_by_division_and_leaves_out_small_diseases(tmp_path, monkeypatch):
    monkeypatch.setattr(tycho, "MIN_WINDOWS", 30)
    rows = []
    for k in range(100):                                       # 100 consecutive epi weeks of 1950-51
        y, w = (1950, k + 1) if k < 52 else (1951, k - 51)
        ew = y * 100 + w
        for st in ("NJ", "NY", "PA", "IL", "IN", "MI"):        # mid-Atlantic (val) and east-north-central (train)
            rows.append([ew, st, st, "STATE", "MEASLES", 5, 1.5])
        for st in ("IL", "IN", "MI") if k < 70 else ():        # a disease with only 7 windows
            rows.append([ew, st, st, "STATE", "RUBELLA", 1, 0.2])
        rows.append([ew, "NY", "NEW YORK CITY", "CITY", "DIPHTHERIA", 1, 0.1])   # city rows are ignored
    _write_csv(tmp_path, rows)
    s = tycho.build(str(tmp_path))
    assert s["measles"]["train"] == 37 and s["measles"]["val"] == 37          # 100 - 64 + 1 each
    assert "left_out" in s["rubella"] and "diphtheria" not in s
    loaded = tycho.load(str(tmp_path), "val")
    assert set(loaded) == {"tycho_measles"} and loaded["tycho_measles"]["x"].shape[0] == 37
    assert json.load(open(tmp_path / "tycho_summary.json"))["measles"]["years"] == [1950, 1950]


def test_fetch_refuses_a_bad_checksum(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(tycho.CSV_NAME, "x")

    class Resp:
        def read(self):
            return buf.getvalue()

    monkeypatch.setattr(tycho.urllib.request, "urlopen", lambda *a, **k: Resp())
    with pytest.raises(SystemExit, match="checksum mismatch"):
        tycho.fetch(str(tmp_path))
    assert not os.path.exists(tmp_path / tycho.CSV_NAME)
