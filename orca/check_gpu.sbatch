#!/bin/bash
# Short GPU health check (~5 min). Submit from the repo root:
#
#     sbatch orca/check_gpu.sbatch
#
# Output goes to orca/logs/lcm-check-<jobid>.out
#SBATCH --job-name=lcm-check
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=orca/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source orca/env.sh

echo "== Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi
echo "== GPU check"
.venv/bin/python orca/gpu_check.py
echo "== Test suite"
.venv/bin/python -m pytest -q
echo "== Done at $(date)"
