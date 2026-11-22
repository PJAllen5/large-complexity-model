#!/bin/bash
# Phase 1 on one real dataset: one array task per model size (index into SIZES).
#     DATASET=flu sbatch --array=0-1 orca/phase1.sbatch          # 8.5M and 28M
# Each task runs the baselines, then for both data regimes (small, full) and
# seeds 0-2: a from-scratch model and one fine-tuned model per pretraining
# variant. PHASE1_VARIANTS lists them as label=folder (default: sims = the
# 160k-step simulator joint runs); each folder holds <size>_seed<N>/checkpoints/joint.pt.
# Tycho experiment:
#   PHASE1_OUT=~/lcm-runs/phase1_tycho PHASE1_SIZES=8.5M \
#   PHASE1_VARIANTS="sims=$HOME/lcm-runs/stageA_joint160k tycho=$HOME/lcm-runs/pretrain_tycho simtycho=$HOME/lcm-runs/pretrain_simtycho" \
#   DATASET=flu sbatch --array=0 orca/phase1.sbatch
# Finished runs are skipped, so resubmitting only does what's left.
#SBATCH --job-name=lcm-phase1
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=orca/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ ! -f phase1.py ] || [ ! -f orca/env.sh ]; then
    echo "!! Submit from the repo root: cd ~/large-complexity-model"; exit 1
fi
source orca/env.sh || exit 1
source orca/stageA_settings.sh                      # for stageA_model_args

DATASET="${DATASET:?set DATASET, e.g. DATASET=flu sbatch ...}"
read -r -a SIZES <<< "${PHASE1_SIZES:-8.5M 28M}"
size="${SIZES[$SLURM_ARRAY_TASK_ID]}"
OUT="${PHASE1_OUT:-$HOME/lcm-runs/phase1_v2}"   # phase1 = the first flu run (coarse validation)
PRETRAINED="${PHASE1_PRETRAINED:-$HOME/lcm-runs/stageA_joint160k}"
read -r -a VARIANTS <<< "${PHASE1_VARIANTS:-sims=$PRETRAINED}"
read -r -a EXTRA_ARGS <<< "${PHASE1_ARGS:-}"        # optional extra phase1.py flags (e.g. for a dry run)
model_args="$(stageA_model_args "$size")"

echo "== Phase 1 $DATASET, $size on $(hostname) at $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
git log --oneline -1 || echo "(git version unavailable)"
for regime in small full; do
    .venv/bin/python -u phase1.py --dataset "$DATASET" --regime "$regime" --mode baselines --out "$OUT"
    for seed in 0 1 2; do
        for v in "${VARIANTS[@]}"; do                  # check every checkpoint before training anything
            init="${v#*=}/${size}_seed${seed}/checkpoints/joint.pt"
            [ -f "$init" ] || { echo "!! missing pretrained checkpoint $init (${v%%=*})"; exit 1; }
        done
        # shellcheck disable=SC2086
        .venv/bin/python -u phase1.py --dataset "$DATASET" --regime "$regime" --mode scratch --seed "$seed" \
            --size "$size" $model_args --precision bf16 --out "$OUT" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        for v in "${VARIANTS[@]}"; do
            # shellcheck disable=SC2086
            .venv/bin/python -u phase1.py --dataset "$DATASET" --regime "$regime" --mode finetune --seed "$seed" \
                --size "$size" $model_args --precision bf16 --out "$OUT" \
                --init "${v#*=}/${size}_seed${seed}/checkpoints/joint.pt" --pretrained "${v%%=*}" \
                ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        done
    done
done
echo "== Done at $(date)"
