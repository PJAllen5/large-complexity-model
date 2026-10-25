"""Training loop: next-patch likelihood on a mix of domains."""

import math
import time
import torch

from .data import MixedSampler, normalize


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(model, datasets, steps, batch_size=64, lr=3e-4, seed=0, warmup=200, log_every=500, tag=""):
    """Train `model` on `datasets` ({name: tensors}) for `steps` updates. Returns loss history."""
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
        loss = model.loss(xn, b["ent_mask"], b["feat_mask"], b["adj"], stats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        if step % log_every == 0 or step == steps:
            n = log_every if step % log_every == 0 else step % log_every
            hist.append((step, run / n))
            print(f"  [{tag}] step {step:5d}  nll {run / n:7.3f}  {time.time() - t0:6.0f}s", flush=True)
            run = 0.0
    model.eval()
    return hist
