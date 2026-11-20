"""Training loop: next-patch likelihood on a mix of domains."""

import contextlib
import math
import time
import torch

from .data import MixedSampler, normalize

PRECISIONS = ("fp32", "tf32", "bf16")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_precision(precision):
    """fp32: exact 32-bit maths (Phase 0 default).
    tf32: NVIDIA tensor cores for matrix multiplies (~10-bit mantissa inputs, fp32 results).
    bf16: transformer blocks in bfloat16 via autocast; the forecast head stays fp32.
    tf32 and bf16 only change anything on NVIDIA GPUs."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    fast = precision in ("tf32", "bf16")
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast
    torch.set_float32_matmul_precision("high" if fast else "highest")


def autocast_for(precision, device):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def validation_nll(model, val_sets, precision="fp32", batch_size=64):
    """Average next-patch NLL on held-out validation episodes, each domain weighted equally."""
    was_training = model.training
    model.eval()
    per_domain = []
    for d in val_sets.values():
        dev = d["x"].device
        losses, counts = 0.0, 0
        for i in range(0, len(d["x"]), batch_size):
            b = {k: v[i:i + batch_size] for k, v in d.items()}
            xn, stats = normalize(b["x"], b["ent_mask"], b["feat_mask"])
            with autocast_for(precision, dev):
                loss = model.loss(xn, b["ent_mask"], b["feat_mask"], b["adj"], stats)
            losses += loss.item() * len(b["x"])
            counts += len(b["x"])
        per_domain.append(losses / counts)
    model.train(was_training)
    return sum(per_domain) / len(per_domain)


def train(model, datasets, steps, batch_size=64, lr=3e-4, seed=0, warmup=200, log_every=500, tag="",
          val=None, precision="fp32"):
    """Train `model` on `datasets` ({name: tensors}) for `steps` updates.

    Returns the history as (step, train_nll) tuples, or (step, train_nll, val_nll)
    when `val` ({name: tensors}) is given. A validation NLL that rises while the
    training NLL keeps falling means the model is overfitting (memorising).
    """
    set_precision(precision)
    sampler = MixedSampler(datasets)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    model.train()
    hist, t0, run = [], time.time(), 0.0
    for step in range(1, steps + 1):
        b = sampler(batch_size, gen)
        xn, stats = normalize(b["x"], b["ent_mask"], b["feat_mask"])
        with autocast_for(precision, xn.device):
            loss = model.loss(xn, b["ent_mask"], b["feat_mask"], b["adj"], stats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            msg = f"  [{tag}] step {step:5d}  nll {run / n:7.3f}"
            if val:
                v = validation_nll(model, val, precision)
                hist.append((step, run / n, v))
                msg += f"  val {v:7.3f}"
            else:
                hist.append((step, run / n))
            print(f"{msg}  {time.time() - t0:6.0f}s", flush=True)
            run = 0.0
    model.eval()
    return hist
