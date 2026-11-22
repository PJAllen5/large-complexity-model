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
def validation_by_domain(model, val_sets, precision="fp32", batch_size=64):
    """Next-patch NLL on held-out validation episodes, per domain ({name: nll})."""
    was_training = model.training
    model.eval()
    per_domain = {}
    for name, d in val_sets.items():
        dev = d["x"].device
        losses, counts = 0.0, 0
        for i in range(0, len(d["x"]), batch_size):
            b = {k: v[i:i + batch_size] for k, v in d.items()}
            xn, stats = normalize(b["x"], b["ent_mask"], b["feat_mask"])
            with autocast_for(precision, dev):
                loss = model.loss(xn, b["ent_mask"], b["feat_mask"], b["adj"], stats)
            losses += loss.item() * len(b["x"])
            counts += len(b["x"])
        per_domain[name] = losses / counts
    model.train(was_training)
    return per_domain


def validation_nll(model, val_sets, precision="fp32", batch_size=64):
    """Average next-patch NLL on held-out validation episodes, each domain weighted equally."""
    per_domain = validation_by_domain(model, val_sets, precision, batch_size)
    return sum(per_domain.values()) / len(per_domain)


def train(model, datasets, steps, batch_size=64, lr=3e-4, seed=0, warmup=200, log_every=500, tag="",
          val=None, precision="fp32", keep_best=True, weights=None, val_log=None, eval_at=None, patience=None):
    """Train `model` on `datasets` ({name: tensors}) for `steps` updates.

    Returns the history as (step, train_nll) tuples, or (step, train_nll, val_nll)
    when `val` ({name: tensors}) is given. A validation NLL that rises while the
    training NLL keeps falling means the model is overfitting (memorising).

    With `val` and `keep_best`, the model ends with the weights from the logged
    step that had the lowest validation NLL ("early stopping"), not the last step.

    `weights` ({name: w}, optional) draws batches in proportion to w instead of
    equally. `val_log` (a list, optional) receives [step, {name: val_nll}] at every
    logged step, so per-domain validation curves can be inspected afterwards.

    `eval_at` (optional) replaces the even `log_every` spacing with explicit steps,
    e.g. dense early checks for fine-tuning. Step 0 means "before any training":
    the starting weights are validated too and can be the ones kept. The last step
    is always checked.

    `patience` (optional, with `val` and `keep_best`): stop early once this many
    validation checks in a row have not beaten the best so far. The best weights
    are kept as usual; the history then ends at the step where training stopped.
    """
    set_precision(precision)
    sampler = MixedSampler(datasets, weights)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1))))
    model.train()
    hist, t0, run, last = [], time.time(), 0.0, 0
    best_val, best_state, since_best = float("inf"), None, 0
    checks = set(eval_at) if eval_at is not None else None

    def check(step, train_nll):
        nonlocal best_val, best_state, since_best
        msg = f"  [{tag}] step {step:5d}  nll {train_nll:7.3f}"
        if val:
            per = validation_by_domain(model, val, precision)
            v = sum(per.values()) / len(per)
            hist.append((step, train_nll, v))
            if val_log is not None:
                val_log.append([step, per])
            msg += f"  val {v:7.3f}"
            if keep_best and v < best_val:
                best_val, since_best = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                msg += "  *best"
            else:
                since_best += 1
        else:
            hist.append((step, train_nll))
        print(f"{msg}  {time.time() - t0:6.0f}s", flush=True)

    if checks is not None and 0 in checks:
        check(0, float("nan"))
        model.train()
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
        due = step in checks if checks is not None else step % log_every == 0
        if due or step == steps:
            check(step, run / (step - last))
            run, last = 0.0, step
            if patience and val and keep_best and since_best >= patience:
                print(f"  [{tag}] early stop at step {step}: no new best in {patience} checks", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return hist
