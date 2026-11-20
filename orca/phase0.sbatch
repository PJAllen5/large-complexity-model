#!/bin/bash
# Rerun the full Phase 0 experiment on one ORCA GPU, then compare with the
# laptop results. Submit from the repo root:
#
#     sbatch orca/phase0.sbatch                       # on an A30
#     sbatch --gres=gpu:l40s:1 orca/phase0.sbatch     # on an L40S instead
#
# Results go to results_orca/ (report, charts, per-run metrics). The log is in
# orca/logs/lcm-phase0-<jobid>.out. If the job is stopped, submitting it again
# resumes: finished runs are cached in results_orca/runs/.
#SBATCH --job-name=lcm-phase0
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=orca/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment.py ] || [ ! -f orca/env.sh ]; then
    echo "!! Submit this job from the repo root: cd ~/large-complexity-model, then sbatch again."; exit 1
fi
if [ ! -x .venv/bin/python ]; then
    echo "!! No Python environment yet: run 'bash orca/setup.sh' first."; exit 1
fi
source orca/env.sh || exit 1

echo "== Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch, numpy; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| numpy', numpy.__version__)"
git log --oneline -1

start=$(date +%s)
.venv/bin/python -u experiment.py --out results_orca
echo "== Experiment took $(( ($(date +%s) - start) / 60 )) min"

echo "== Comparison with the laptop run"
.venv/bin/python orca/compare.py results results_orca
echo "== Done at $(date)"
