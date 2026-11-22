"""Real datasets in the universal format (Phase 1).

A real dataset is one long recording, `values` [T, N, F] (NaN = missing), plus an
interaction graph `adj` [N, N] and time ranges for each split. Episodes are
64-step windows that lie entirely inside one split and have no missing values,
so the model sees exactly the same format as the simulators: 48 steps of
history, 16 to forecast.

Splits are by time (train earliest, then validation, then test), so a model is
never trained on the period it's tested on.
"""

import os

import numpy as np
import torch

from .data import FMAX, NMAX, T_CTX, T_TOTAL, normalize

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")


def to_episodes(values, adj, starts):
    """Universal-format episodes for the windows starting at `starts` (skipping any with gaps)."""
    T, N, F = values.shape
    if N > NMAX or F > FMAX:
        raise ValueError(f"{N} entities x {F} features exceeds the model's {NMAX} x {FMAX}")
    keep = [s for s in starts if 0 <= s and s + T_TOTAL <= T and np.isfinite(values[s:s + T_TOTAL]).all()]
    x = np.zeros((len(keep), T_TOTAL, NMAX, FMAX), np.float32)
    for i, s in enumerate(keep):
        x[i, :, :N, :F] = values[s:s + T_TOTAL]
    a = np.zeros((len(keep), NMAX, NMAX), np.float32)
    a[:, :N, :N] = adj
    em = np.zeros((len(keep), NMAX), np.float32)
    em[:, :N] = 1
    fm = np.zeros((len(keep), FMAX), np.float32)
    fm[:, :F] = 1
    return dict(x=x, ent_mask=em, feat_mask=fm, adj=a), keep


def split_starts(lo, hi, stride=1):
    """Window starts whose whole 64-step window fits in [lo, hi)."""
    return list(range(lo, hi - T_TOTAL + 1, stride))


def seasonal_episodes(values, adj, starts, period):
    """Same windows, but the 16 forecast steps replaced by the values one `period` earlier.

    Used for the seasonal-naive baseline ("same week last year"). Windows whose
    lagged values are missing or before the start of the recording are dropped;
    returns (episodes, kept starts).
    """
    d, kept = to_episodes(values, adj, starts)
    N, F = values.shape[1:]
    ok = []
    for i, s in enumerate(kept):
        lo, hi = s + T_CTX - period, s + T_TOTAL - period
        if lo >= 0 and np.isfinite(values[lo:hi]).all():
            d["x"][i, T_CTX:, :N, :F] = values[lo:hi]   # each window built separately: windows overlap
            ok.append(i)
    return {k: v[ok] for k, v in d.items()}, [kept[i] for i in ok]


@torch.no_grad()
def point_crps(d, forecast_x=None):
    """CRPS of a point forecast (= mean absolute error), in the model's normalised units.

    Persistence (last observed value) when `forecast_x` is None; otherwise the
    forecast steps of `forecast_x`, which must share d's history so both are
    normalised with the same statistics.
    """
    xn, _ = normalize(d["x"], d["ent_mask"], d["feat_mask"])
    truth = xn[:, T_CTX:]
    if forecast_x is None:
        guess = xn[:, T_CTX - 1:T_CTX].expand_as(truth)
    else:
        assert torch.equal(forecast_x[:, :T_CTX], d["x"][:, :T_CTX]), "forecast must share the history"
        guess = normalize(forecast_x, d["ent_mask"], d["feat_mask"])[0][:, T_CTX:]
    m = (d["ent_mask"][:, None, :, None] * d["feat_mask"][:, None, None, :]).expand_as(truth)
    return float(((guess - truth).abs() * m).sum() / m.sum())


# --- datasets -----------------------------------------------------------------

# US HHS regions that share a land border (1 New England, 2 NY/NJ, 3 Mid-Atlantic,
# 4 Southeast, 5 Great Lakes, 6 South Central, 7 Central Plains, 8 Mountain,
# 9 Pacific/Southwest, 10 Northwest).
HHS_BORDERS = [(1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7), (5, 7), (5, 8), (6, 7),
               (6, 8), (6, 9), (7, 8), (8, 9), (8, 10), (9, 10)]


def epiweek_season(epiweek):
    """Flu season label: weeks 40-53 of year Y and 1-39 of Y+1 belong to season Y."""
    y, w = divmod(epiweek, 100)
    return y if w >= 40 else y - 1


def load_flu(path=os.path.join(DATA_DIR, "flu_hhs_wili.csv")):
    """CDC ILINet weighted % of doctor visits for influenza-like illness, 10 HHS regions, weekly.

    Splits by flu season (Phase 1 plan, PROJECT_LOG.md 2026-11-21):
      train 2002-03 to 2012-13 (11 seasons); small-data train: the last 3 of those
      val   2013-14 and 2014-15
      test  2015-16 onward (includes the 2020-21 COVID season with almost no flu)
    """
    raw = np.genfromtxt(path, delimiter=",", names=True)
    weeks = raw["epiweek"].astype(int)
    if np.any(np.diff(weeks) <= 0):
        raise ValueError("epiweeks must be strictly increasing")
    values = np.stack([raw[f"hhs{i}"] for i in range(1, 11)], 1)[:, :, None].astype(np.float32)
    adj = np.zeros((10, 10), np.float32)
    for a, b in HHS_BORDERS:
        adj[a - 1, b - 1] = adj[b - 1, a - 1] = 1
    season = np.array([epiweek_season(w) for w in weeks])
    first = lambda s: int(np.argmax(season >= s))   # noqa: E731  index of a season's first week
    splits = dict(train=(0, first(2013)), small_train=(first(2010), first(2013)),
                  val=(first(2013), first(2015)), test=(first(2015), len(weeks)))
    return dict(name="flu", values=values, adj=adj, times=weeks, period=52, splits=splits,
                units="% of doctor visits with influenza-like illness (weighted)")


REAL = {"flu": load_flu}
