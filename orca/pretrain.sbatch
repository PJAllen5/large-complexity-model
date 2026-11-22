#!/bin/bash
# Real-data pretraining for the Tycho experiment: 6 array tasks = 2 variants x 3 seeds.
#     sbatch --array=0-5 orca/pretrain.sbatch
#   0-2: tycho    (Tycho only)            -> ~/lcm-runs/pretrain_tycho/8.5M_seed<N>
#   3-5: simtycho (simulators + Tycho)    -> ~/lcm-runs/pretrain_simtycho/8.5M_seed<N>
# First build the Tycho windows on the login node (needs internet, ~10 s):
#     .venv/bin/python -m lcm.tycho fetch ~/lcm-runs/tycho && .venv/bin/python -m lcm.tycho build ~/lcm-runs/tycho
# Finished runs (checkpoint present) are skipped.
#SBATCH --job-name=lcm-pretrain
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=orca/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ ! -f pretrain.py ] || [ ! -f orca/env.sh ]; then
    echo "!! Submit from the repo root: cd ~/large-complexity-model"; exit 1
fi
source orca/env.sh || exit 1
source orca/stageA_settings.sh                      # for stageA_model_args

VARIANTS=(tycho simtycho)
variant="${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed=$((SLURM_ARRAY_TASK_ID % 3))
size=8.5M
TYCHO="${PRETRAIN_TYCHO:-$HOME/lcm-runs/tycho}"
SIM_DATA="${PRETRAIN_SIM_DATA:-$HOME/lcm-runs/stageA_n20k/data}"
OUT="${PRETRAIN_ROOT:-$HOME/lcm-runs}/pretrain_${variant}/${size}_seed${seed}"
read -r -a EXTRA_ARGS <<< "${PRETRAIN_ARGS:-}"      # optional extra pretrain.py flags (e.g. for a dry run)
[ -f "$TYCHO/tycho_summary.json" ] || { echo "!! no Tycho windows in $TYCHO; build them first (see top)"; exit 1; }

echo "== Pretrain $variant seed $seed ($size) on $(hostname) at $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
git log --oneline -1 || echo "(git version unavailable)"
# shellcheck disable=SC2046
.venv/bin/python -u pretrain.py --variant "$variant" --seed "$seed" $(stageA_model_args "$size") --precision bf16 \
    --tycho "$TYCHO" --sim-data "$SIM_DATA" --out "$OUT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
echo "== Done at $(date)"
