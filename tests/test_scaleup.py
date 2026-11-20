"""Checks for the scale-up options: model sizes, precision, validation and data splits."""

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
        hist = train(m, sets, 10, batch_size=4, seed=0, log_every=5, val=v)
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
