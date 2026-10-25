"""The universal data format: every system becomes entities x time x features.

An episode is stored as
    x         [T, NMAX, FMAX]  observations, zero-padded
    ent_mask  [NMAX]           1 where the entity exists
    feat_mask [FMAX]           1 where the feature exists
    adj       [NMAX, NMAX]     interaction graph

Values are normalised per episode using only the context window (the past the
model is allowed to see), so a whale track and a galaxy look alike in scale and
nothing leaks from the future. An asinh squash keeps sudden take-offs finite.
"""

import os
import numpy as np
import torch

from .simulators import SIMULATORS, DOMAINS

NMAX, FMAX = 12, 4
T_TOTAL, T_CTX, PATCH = 64, 48, 4
HORIZON = T_TOTAL - T_CTX


def pad_episodes(sim, obs, adj):
    B, T, N, F = obs.shape
    x = np.zeros((B, T, NMAX, FMAX), np.float32)
    x[:, :, :N, :F] = obs
    a = np.zeros((B, NMAX, NMAX), np.float32)
    a[:, :N, :N] = adj
    em = np.zeros((B, NMAX), np.float32)
    em[:, :N] = 1
    fm = np.zeros((B, FMAX), np.float32)
    fm[:, :F] = 1
    return dict(x=x, ent_mask=em, feat_mask=fm, adj=a)


def generate(name, n_episodes, seed, chunk=250):
    """Simulate n_episodes of one system in the universal format."""
    sim = SIMULATORS[name]
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, n_episodes, chunk):
        B = min(chunk, n_episodes - start)
        state, params = sim.init(rng, B)
        state, _ = sim.run(state, params, rng, int(rng.integers(0, sim.max_burn + 1)))
        _, obs = sim.run(state, params, rng, T_TOTAL)
        parts.append(pad_episodes(sim, obs, params["adj"]))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["x"] = np.nan_to_num(out["x"], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_datasets(data_dir, n_train, n_test, seed=0):
    os.makedirs(data_dir, exist_ok=True)
    for i, name in enumerate(DOMAINS):
        for split, n, s in (("train", n_train, seed + 100 * i), ("test", n_test, seed + 100 * i + 50)):
            path = os.path.join(data_dir, f"{name}_{split}.npz")
            if not os.path.exists(path):
                np.savez_compressed(path, **generate(name, n, s))


def load(data_dir, name, split, device, limit=None):
    d = np.load(os.path.join(data_dir, f"{name}_{split}.npz"))
    return {k: torch.from_numpy(d[k][:limit]).to(device) for k in d.files}


def normalize(x, ent_mask, feat_mask, t_ctx=T_CTX):
    """Per-episode, per-feature scaling using the context window only.

    Returns normalised x and stats [B, 2*FMAX] (mean, log std) the model can
    use to know the original scale.
    """
    ctx = x[:, :t_ctx]
    w = ent_mask[:, None, :, None].expand_as(ctx)
    n = w.sum((1, 2)).clamp(min=1)
    mean = (ctx * w).sum((1, 2)) / n
    var = (((ctx - mean[:, None, None]) ** 2) * w).sum((1, 2)) / n
    std = var.sqrt().clamp(min=1e-6)
    # asinh: linear near 0, logarithmic for big departures (an epidemic taking off
    # from a flat start), so nothing needs to be clipped.
    xn = torch.asinh((x - mean[:, None, None]) / std[:, None, None])
    xn = xn * ent_mask[:, None, :, None] * feat_mask[:, None, None, :]
    stats = torch.cat([mean, std.log()], -1) * feat_mask.repeat(1, 2)
    return xn, stats


class MixedSampler:
    """Draws training batches with every domain equally represented."""

    def __init__(self, datasets):
        self.datasets = datasets

    def __call__(self, batch_size, gen):
        names = list(self.datasets)
        which = torch.randint(len(names), (batch_size,), generator=gen)
        parts = []
        for i, name in enumerate(names):
            k = int((which == i).sum())
            if k == 0:
                continue
            d = self.datasets[name]
            idx = torch.randint(len(d["x"]), (k,), generator=gen).to(d["x"].device)
            parts.append({key: v[idx] for key, v in d.items()})
        return {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
