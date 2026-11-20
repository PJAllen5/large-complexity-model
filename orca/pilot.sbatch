#!/bin/bash
# Scale-up pilot: speed, memory and precision check for each model size.
# Submit from the repo root, once per GPU type:
#
#     sbatch orca/pilot.sbatch                          # A30
#     sbatch --gres=gpu:l40s:1 orca/pilot.sbatch        # L40S
#
# Log: orca/logs/lcm-pilot-<jobid>.out
#SBATCH --job-name=lcm-pilot
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
git log --oneline -1 || echo "(git version unavailable)"
start=$(date +%s)
.venv/bin/python -u orca/pilot.py
echo "== Pilot took $(( ($(date +%s) - start) / 60 )) min, done at $(date)"
