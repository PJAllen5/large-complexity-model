"""Fast checks of the LCM's data pipeline and model. All run on CPU in under a minute.

They don't judge forecast quality (that's experiment.py's job). They catch the bugs
that would quietly make an experiment meaningless: wrong shapes, future data leaking
into the past, missing entities influencing real ones, or a model that can't learn.
"""

import numpy as np
import pytest
import torch

from lcm.data import FMAX, NMAX, PATCH, T_CTX, T_TOTAL, HORIZON, MixedSampler, generate, normalize
from lcm.model import LCM
from lcm.simulators import DOMAINS, SIMULATORS
from lcm.train import train

torch.set_num_threads(2)


def _batch(name="predator_prey", n=4, seed=0):
    d = {k: torch.from_numpy(v) for k, v in generate(name, n, seed).items()}
    xn, stats = normalize(d["x"], d["ent_mask"], d["feat_mask"])
    return d, xn, stats


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return LCM().eval()


# ---------------------------------------------------------------- data

@pytest.mark.parametrize("name", DOMAINS)
def test_every_simulator_produces_valid_episodes(name):
    sim = SIMULATORS[name]
    d = generate(name, 3, seed=1)
    assert d["x"].shape == (3, T_TOTAL, NMAX, FMAX)
    assert np.isfinite(d["x"]).all()
    assert (d["ent_mask"].sum(1) == sim.n).all()
    assert (d["feat_mask"].sum(1) == sim.f).all()
    # padding (entities / features the system doesn't have) stays zero
    assert not d["x"][:, :, sim.n:].any() and not d["x"][:, :, :, sim.f:].any()
    # the interaction graph is symmetric with no self-links
    assert np.allclose(d["adj"], d["adj"].transpose(0, 2, 1))
    assert not np.diagonal(d["adj"], axis1=1, axis2=2).any()


def test_generation_is_reproducible():
    a, b = generate("boids", 2, seed=7), generate("boids", 2, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_normalize_does_not_peek_at_the_future():
    d, xn, stats = _batch()
    x2 = d["x"].clone()
    x2[:, T_CTX:] = x2[:, T_CTX:] * 100 + 50          # change only the future
    xn2, stats2 = normalize(x2, d["ent_mask"], d["feat_mask"])
    assert torch.equal(stats, stats2)
    assert torch.equal(xn[:, :T_CTX], xn2[:, :T_CTX])


def test_mixed_sampler_draws_from_every_domain():
    data = {n: {k: torch.from_numpy(v) for k, v in generate(n, 4, 0).items()} for n in DOMAINS[:3]}
    b = MixedSampler(data)(30, torch.Generator().manual_seed(0))
    assert b["x"].shape[0] == 30
    n_entities = set(b["ent_mask"].sum(1).int().tolist())
    assert n_entities == {SIMULATORS[n].n for n in DOMAINS[:3]}


# ---------------------------------------------------------------- model

def test_parameter_count_matches_project_log(model):
    # PROJECT_LOG.md and README.md say 1.08M. Update them if this changes.
    assert sum(p.numel() for p in model.parameters()) == 1_081_380


def test_forward_shapes_and_finite_loss(model):
    d, xn, stats = _batch()
    logits, mu, logsig = model(xn, d["ent_mask"], d["feat_mask"], d["adj"], stats)
    Tp = T_TOTAL // PATCH
    assert logits.shape == (4, NMAX, Tp, model.K)
    assert mu.shape == logsig.shape == (4, NMAX, Tp, model.K, PATCH * FMAX)
    loss = model.loss(xn, d["ent_mask"], d["feat_mask"], d["adj"], stats)
    assert torch.isfinite(loss)


def test_gradients_are_finite():
    torch.manual_seed(0)
    m = LCM()
    d, xn, stats = _batch()
    m.loss(xn, d["ent_mask"], d["feat_mask"], d["adj"], stats).backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_predictions_are_causal_in_time(model):
    """A prediction made at time patch t must not change when later patches change."""
    d, xn, stats = _batch()
    xn2 = xn.clone()
    xn2[:, -PATCH:] += 3.0                             # change only the last patch
    args = (d["ent_mask"], d["feat_mask"], d["adj"], stats)
    _, mu1, _ = model(xn, *args)
    _, mu2, _ = model(xn2, *args)
    assert torch.allclose(mu1[:, :, :-1], mu2[:, :, :-1], atol=1e-5)
    assert not torch.allclose(mu1[:, :, -1], mu2[:, :, -1])


def test_missing_entities_do_not_influence_real_ones(model):
    """predator_prey has 6 of 12 entity slots; junk in the empty slots must be ignored."""
    d, xn, stats = _batch("predator_prey")
    n = SIMULATORS["predator_prey"].n
    xn2 = xn.clone()
    xn2[:, :, n:] = torch.randn_like(xn2[:, :, n:]) * 5
    args = (d["ent_mask"], d["feat_mask"], d["adj"], stats)
    _, mu1, _ = model(xn, *args)
    _, mu2, _ = model(xn2, *args)
    assert torch.allclose(mu1[:, :n], mu2[:, :n], atol=1e-5)


def test_sample_future_shape(model):
    d, xn, stats = _batch(n=2)
    torch.manual_seed(0)
    fut = model.sample_future(xn[:, :T_CTX], d["ent_mask"], d["feat_mask"], d["adj"], stats,
                              HORIZON, n_samples=3)
    assert fut.shape == (2, 3, HORIZON, NMAX, FMAX)
    assert torch.isfinite(fut).all()


def test_short_training_lowers_the_loss():
    torch.manual_seed(0)
    m = LCM(d=32, layers=1, heads=2)
    data = {n: {k: torch.from_numpy(v) for k, v in generate(n, 32, 0).items()} for n in DOMAINS[:2]}
    hist = train(m, data, steps=60, batch_size=16, lr=3e-3, warmup=5, log_every=20, tag="test")
    assert hist[-1][1] < hist[0][1]
