#!/bin/bash
# Measure run-to-run noise: rerun Phase 0 with a different training seed on
# ORCA, then compare with the first ORCA run (seed 0, results_orca/). Same
# hardware and same data, so the differences come only from training chance.
# Submit from the repo root:
#
#     sbatch orca/seed_noise.sbatch                          # seed 1
#     sbatch --export=ALL,SEED=2 orca/seed_noise.sbatch      # another seed
#
# Results go OUTSIDE the repo, to ~/lcm-runs/phase0_seed<SEED>/, so a later
# `git pull` never collides with them. Log: orca/logs/lcm-seed-<jobid>.out
#SBATCH --job-name=lcm-seed
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
if [ ! -d results_orca/runs ]; then
    echo "!! results_orca/runs is missing: run 'git pull' (it holds the seed-0 ORCA run)."; exit 1
fi
source orca/env.sh || exit 1

SEED="${SEED:-1}"
OUT="$HOME/lcm-runs/phase0_seed$SEED"
mkdir -p "$HOME/lcm-runs"

echo "== Job $SLURM_JOB_ID on $(hostname) at $(date) | seed $SEED | output $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python -c "import torch, numpy; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| numpy', numpy.__version__)"
git log --oneline -1

start=$(date +%s)
.venv/bin/python -u experiment.py --seed "$SEED" --out "$OUT"
echo "== Experiment took $(( ($(date +%s) - start) / 60 )) min"

echo "== Noise: seed $SEED vs seed 0 (both on ORCA)"
.venv/bin/python orca/compare.py --noise results_orca "$OUT"
echo "== Done at $(date)"
