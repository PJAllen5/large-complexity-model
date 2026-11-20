"""Quick health check of a GPU node: is the GPU visible to PyTorch, does it
compute correctly, and how fast is one LCM training step?

Runs inside orca/check_gpu.sbatch. For reference, one step took 0.065 s on
the M3 MacBook Air used for Phase 0.
"""

import sys
import time

import torch

from lcm.model import LCM
from lcm.train import get_device

print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FAIL: PyTorch can't see a GPU. See 'Troubleshooting' in orca/README.md.")
    sys.exit(1)

dev = get_device()
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | {props.total_memory / 1e9:.0f} GB | device used by LCM: {dev}")

# correctness: GPU matrix multiply matches CPU
a, b = torch.randn(512, 512), torch.randn(512, 512)
err = (a @ b - (a.to(dev) @ b.to(dev)).cpu()).abs().max().item()
print(f"matmul GPU vs CPU max difference: {err:.2e} ({'ok' if err < 1e-2 else 'SUSPICIOUS'})")

# speed: same model size and batch as the Phase 0 run
B = 32
m = LCM().to(dev)
opt = torch.optim.AdamW(m.parameters())
x = torch.randn(B, 64, 12, 4, device=dev)
em, fm = torch.ones(B, 12, device=dev), torch.ones(B, 4, device=dev)
adj, st = torch.ones(B, 12, 12, device=dev), torch.zeros(B, 8, device=dev)
for i in range(25):
    if i == 5:
        torch.cuda.synchronize()
        t = time.time()
    loss = m.loss(x, em, fm, adj, st)
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
step = (time.time() - t) / 20
print(f"training step: {step:.4f} s (M3 laptop: 0.065 s -> {0.065 / step:.1f}x faster)")
print("PASS")
