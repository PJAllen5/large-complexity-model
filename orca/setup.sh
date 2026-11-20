#!/bin/bash
# One-time setup on ORCA. Run it on the LOGIN node from the repo root:
#
#     bash orca/setup.sh
#
# It only creates the Python environment and installs packages (light work,
# allowed on the login node). It trains nothing. Takes ~5-10 minutes, mostly
# downloading PyTorch (~3 GB, which goes in your 100 GB /home).
set -euo pipefail
cd "$(dirname "$0")/.."

source orca/env.sh            # loads ORCA's Python module

echo "== Python: $(python3 --version)"
if [ -d .venv ]; then
    echo "== .venv already exists; reusing it (delete it to start fresh)"
else
    python3 -m venv .venv
fi

.venv/bin/pip install --no-cache-dir --upgrade pip
# Default Linux PyTorch build = CUDA version. If the GPU check later says
# CUDA isn't available, see "Troubleshooting" in orca/README.md.
.venv/bin/pip install --no-cache-dir torch
.venv/bin/pip install --no-cache-dir -r requirements-dev.txt

mkdir -p orca/logs

.venv/bin/python -c "import torch, numpy; print('== torch', torch.__version__, '| built for CUDA', torch.version.cuda, '| numpy', numpy.__version__)"
echo "== Setup done. Next: sbatch orca/check_gpu.sbatch"
