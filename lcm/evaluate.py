"""How good are the model's probabilistic forecasts?

forecast_metrics: on held-out episodes, sample many futures and score them.
    crps      - accuracy of the whole predicted distribution (lower is better).
                For a single-number forecast it equals the average error.
    mse       - error of the average predicted future.
    cover90   - how often the truth lands inside the 90% range (ideal: 0.90).
    persist_* - the "nothing changes" baseline: repeat the last observation.

outcome_likelihood: the direct test of "predicts likelihoods of outcomes".
    Run a system up to a moment, then replay its future many times with fresh
    randomness to get the TRUE odds that a quantity ends up higher than now.
    Compare with the odds the model gives from its sampled futures.
"""

import numpy as np
import torch

from .data import T_CTX, HORIZON, normalize, pad_episodes
from .simulators import SIMULATORS, repeat_tree


def _crps(samples, y):
    """samples [..., S], y [...]. Sample-based CRPS."""
    S = samples.shape[-1]
    s, _ = samples.sort(-1)
    term1 = (s - y[..., None]).abs().mean(-1)
    i = torch.arange(1, S + 1, device=s.device, dtype=s.dtype)
    term2 = ((2 * i - S - 1) * s).sum(-1) / (S * S)
    return term1 - term2


@torch.no_grad()
def forecast_metrics(model, d, n_samples=32, batch=16, limit=256, per_window=False):
    """Average forecast scores over the first `limit` episodes. With `per_window`, also
    returns each episode's CRPS and 90% coverage (lists, in episode order)."""
    sums = dict(crps=0.0, mse=0.0, cover90=0.0, persist_crps=0.0, persist_mse=0.0)
    count = 0.0
    win_crps, win_cover = [], []
    n = min(limit, len(d["x"]))
    for i in range(0, n, batch):
        b = {k: v[i:i + batch] for k, v in d.items()}
        xn, stats = normalize(b["x"], b["ent_mask"], b["feat_mask"])
        ctx, truth = xn[:, :T_CTX], xn[:, T_CTX:]
        fut = model.sample_future(ctx, b["ent_mask"], b["feat_mask"], b["adj"], stats, HORIZON, n_samples)
        fut = fut.permute(0, 2, 3, 4, 1)                              # [B, H, N, F, S]
        m = (b["ent_mask"][:, None, :, None] * b["feat_mask"][:, None, None, :]).expand_as(truth)
        q = torch.quantile(fut, torch.tensor([0.05, 0.95], device=fut.device), dim=-1)
        last = ctx[:, -1:].expand_as(truth)
        c = _crps(fut, truth) * m
        inside = ((truth >= q[0]) & (truth <= q[1])).float() * m
        if per_window:
            per = m.sum((1, 2, 3))
            win_crps += (c.sum((1, 2, 3)) / per).tolist()
            win_cover += (inside.sum((1, 2, 3)) / per).tolist()
        sums["crps"] += c.sum().item()
        sums["mse"] += (((fut.mean(-1) - truth) ** 2) * m).sum().item()
        sums["cover90"] += inside.sum().item()
        sums["persist_crps"] += ((last - truth).abs() * m).sum().item()
        sums["persist_mse"] += (((last - truth) ** 2) * m).sum().item()
        count += m.sum().item()
    out = {k: v / count for k, v in sums.items()}
    if per_window:
        out.update(window_crps=win_crps, window_cover90=win_cover)
    return out


def _to_torch(d, device):
    return {k: torch.from_numpy(v).to(device) for k, v in d.items()}


@torch.no_grad()
def outcome_likelihood(model, name, device, n_ctx=64, n_branch=64, n_samples=64, seed=999, batch=8):
    """Model odds vs true odds that entity 0's key variable rises over the horizon."""
    sim = SIMULATORS[name]
    rng = np.random.default_rng(seed)
    state, params = sim.init(rng, n_ctx)
    state, _ = sim.run(state, params, rng, int(rng.integers(0, sim.max_burn + 1)))
    state, ctx_obs = sim.run(state, params, rng, T_CTX)
    _, fut_obs = sim.run(repeat_tree(state, n_branch), repeat_tree(params, n_branch), rng, HORIZON)

    ctx_full = np.concatenate([ctx_obs, np.zeros((n_ctx, HORIZON) + ctx_obs.shape[2:])], 1)
    ctx_d = _to_torch(pad_episodes(sim, np.nan_to_num(ctx_full), params["adj"]), device)
    br_full = np.concatenate([np.repeat(ctx_obs, n_branch, 0), fut_obs], 1)
    br_d = _to_torch(pad_episodes(sim, np.nan_to_num(br_full), repeat_tree(params, n_branch)["adj"]), device)

    f, margin = sim.event_f, 0.05
    xb, _ = normalize(br_d["x"], br_d["ent_mask"], br_d["feat_mask"])
    rises = (xb[:, -1, 0, f] > xb[:, T_CTX - 1, 0, f] + margin).float()
    p_true = rises.view(n_ctx, n_branch).mean(1)

    xn, stats = normalize(ctx_d["x"], ctx_d["ent_mask"], ctx_d["feat_mask"])
    p_model = []
    for i in range(0, n_ctx, batch):
        s = slice(i, i + batch)
        fut = model.sample_future(xn[s, :T_CTX], ctx_d["ent_mask"][s], ctx_d["feat_mask"][s],
                                  ctx_d["adj"][s], stats[s], HORIZON, n_samples)
        now = xn[s, T_CTX - 1, 0, f][:, None]
        p_model.append((fut[:, :, -1, 0, f] > now + margin).float().mean(1))
    p_model = torch.cat(p_model)
    p_true, p_model = p_true.cpu().numpy(), p_model.cpu().numpy()
    base = p_true.mean()
    return dict(
        prob_error=float(np.abs(p_model - p_true).mean()),
        prob_error_baserate=float(np.abs(base - p_true).mean()),
        p_true=p_true.tolist(), p_model=p_model.tolist(),
    )
