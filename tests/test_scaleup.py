"""Checks for the scale-up options: model sizes, precision, validation and data splits."""

import math

import numpy as np
import pytest
import torch

from lcm.data import build_datasets, generate, load, normalize
from lcm.model import LCM
from lcm.train import set_precision, train, validation_nll

torch.set_num_threads(2)


def _sets(n=8, seed=0, names=("boids", "epidemic")):
    return {name: {k: torch.from_numpy(v) for k, v in generate(name, n, seed + i).items()}
            for i, name in enumerate(names)}


@pytest.mark.parametrize("kw, expected", [(dict(d=256, layers=8, heads=8), 8.47e6),
                                          (dict(d=384, layers=12, heads=8), 28.47e6)])
def test_bigger_sizes_build_and_run(kw, expected):
    m = LCM(**kw)
    assert sum(p.numel() for p in m.parameters()) == pytest.approx(expected, rel=0.01)
    d = _sets(n=2)["boids"]
    xn, st = normalize(d["x"], d["ent_mask"], d["feat_mask"])
    assert torch.isfinite(m.loss(xn, d["ent_mask"], d["feat_mask"], d["adj"], st))


def test_bf16_keeps_forecast_head_in_fp32():
    torch.manual_seed(0)
    m = LCM()
    d = _sets(n=2)["epidemic"]
    xn, st = normalize(d["x"], d["ent_mask"], d["feat_mask"])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, mu, logsig = m(xn, d["ent_mask"], d["feat_mask"], d["adj"], st)
        loss = m.loss(xn, d["ent_mask"], d["feat_mask"], d["adj"], st)
    assert mu.dtype == logsig.dtype == logits.dtype == torch.float32
    exact = m.loss(xn, d["ent_mask"], d["feat_mask"], d["adj"], st)
    assert torch.isfinite(loss) and abs(loss.item() - exact.item()) < 0.1


def test_validation_does_not_change_training():
    sets, val = _sets(), _sets(seed=50)
    weights = []
    for v in (None, val):
        torch.manual_seed(0)
        m = LCM(d=64, layers=2, heads=2)
        hist = train(m, sets, 10, batch_size=4, seed=0, log_every=5, val=v, keep_best=False)
        weights.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
    assert torch.equal(weights[0], weights[1])
    assert len(hist[0]) == 3 and np.isfinite(hist[-1][2])
    assert validation_nll(m, val) == pytest.approx(hist[-1][2], rel=1e-5)


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError):
        set_precision("fp16")
    set_precision("fp32")


def test_validation_split_is_separate_and_rebuilt_on_size_change(tmp_path):
    build_datasets(str(tmp_path), n_train=6, n_test=4, n_val=5)
    tr, va = (load(str(tmp_path), "boids", s, "cpu") for s in ("train", "val"))
    assert len(va["x"]) == 5 and not torch.equal(tr["x"][:5], va["x"])
    build_datasets(str(tmp_path), n_train=6, n_test=4, n_val=7)
    assert len(load(str(tmp_path), "boids", "val", "cpu")["x"]) == 7


def test_keep_best_restores_lowest_validation_weights(monkeypatch):
    """Validation scores dip at the 2nd check, then rise: the model must end with the 2nd check's weights."""
    import lcm.train as T
    scores, snapshots = iter([1.0, 0.5, 0.9, 1.2]), []

    def fake_validation(model, val, precision="fp32"):
        snapshots.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
        return {"unused": next(scores)}

    monkeypatch.setattr(T, "validation_by_domain", fake_validation)
    torch.manual_seed(0)
    m = LCM(d=64, layers=2, heads=2)
    hist = T.train(m, _sets(), 8, batch_size=4, seed=0, log_every=2, val={"unused": None})
    final = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert [h[2] for h in hist] == [1.0, 0.5, 0.9, 1.2]
    assert torch.equal(final, snapshots[1]) and not torch.equal(final, snapshots[3])


def _counts(sampler, n_batches=200, batch=50):
    """How many sampled episodes came from each domain (matched by content)."""
    gen = torch.Generator().manual_seed(0)
    out = dict.fromkeys(sampler.datasets, 0)
    for _ in range(n_batches):
        b = sampler(batch, gen)
        for n, d in sampler.datasets.items():
            out[n] += sum(any(torch.equal(x, y) for y in d["x"]) for x in b["x"])
    assert sum(out.values()) == n_batches * batch
    return out


def test_weighted_sampler_draws_in_proportion():
    from lcm.data import MixedSampler
    sets = _sets(n=4)
    c = _counts(MixedSampler(sets, {"epidemic": 3.0}), n_batches=40)
    assert 2.4 < c["epidemic"] / c["boids"] < 3.7                  # expected 3.0


def test_unweighted_sampler_is_unchanged():
    """No weights must give exactly the old random draws, so earlier runs stay reproducible."""
    from lcm.data import MixedSampler
    sets = _sets(n=4)
    gen = torch.Generator().manual_seed(1)
    which = torch.randint(2, (16,), generator=gen)               # the old implementation's first draw
    expected_epidemic = int((which == 1).sum())
    b = MixedSampler(sets)(16, torch.Generator().manual_seed(1))
    n_epi = sum(any(torch.equal(x, y) for y in sets["epidemic"]["x"]) for x in b["x"])
    assert n_epi == expected_epidemic


def test_bad_weights_are_refused():
    from lcm.data import MixedSampler
    with pytest.raises(ValueError):
        MixedSampler(_sets(n=2), {"nbody": 2.0})
    with pytest.raises(ValueError):
        MixedSampler(_sets(n=2), {"boids": 0.0})


def test_val_log_records_every_domain():
    import lcm.train as T
    torch.manual_seed(0)
    sets = _sets(n=4)
    log = []
    hist = T.train(LCM(d=32, layers=1, heads=2), sets, 4, batch_size=4, seed=0, log_every=2,
                   val=_sets(n=2, seed=50), val_log=log)
    assert [s for s, _ in log] == [2, 4] and set(log[0][1]) == {"boids", "epidemic"}
    assert hist[0][2] == pytest.approx(sum(log[0][1].values()) / 2)   # the average is still what keep_best uses


def test_eval_at_checks_listed_steps_and_step_zero(monkeypatch):
    """Dense early checks; step 0 (the starting weights) can be the one kept."""
    import lcm.train as T
    scores = iter([0.1, 0.5, 0.6, 0.7, 0.8])               # step 0 is best

    monkeypatch.setattr(T, "validation_by_domain", lambda model, val, precision="fp32": {"x": next(scores)})
    torch.manual_seed(0)
    m = LCM(d=32, layers=1, heads=2)
    start = torch.cat([p.detach().flatten().clone() for p in m.parameters()])
    hist = T.train(m, _sets(), 8, batch_size=4, seed=0, val={"x": None}, eval_at=[0, 1, 2, 4])
    assert [h[0] for h in hist] == [0, 1, 2, 4, 8]          # the last step is always checked
    assert torch.equal(torch.cat([p.detach().flatten() for p in m.parameters()]), start)
    assert math.isnan(hist[0][1]) and not math.isnan(hist[2][1])   # step 0 has no training loss yet


def test_default_schedule_is_unchanged():
    import lcm.train as T
    torch.manual_seed(0)
    hist = T.train(LCM(d=32, layers=1, heads=2), _sets(), 7, batch_size=4, seed=0, log_every=3)
    assert [h[0] for h in hist] == [3, 6, 7]


def test_per_window_metrics_average_to_the_totals():
    from lcm.evaluate import forecast_metrics
    from lcm.data import T_TOTAL
    torch.manual_seed(0)
    d = {k: v for k, v in _sets(n=6)["boids"].items()}
    assert d["x"].shape[1] == T_TOTAL
    m = forecast_metrics(LCM(d=32, layers=1, heads=2).eval(), d, n_samples=8, batch=4, limit=6, per_window=True)
    assert len(m["window_crps"]) == 6 and len(m["window_cover90"]) == 6
    assert sum(m["window_crps"]) / 6 == pytest.approx(m["crps"], rel=1e-5)   # equal-sized windows
    assert sum(m["window_cover90"]) / 6 == pytest.approx(m["cover90"], rel=1e-5)
