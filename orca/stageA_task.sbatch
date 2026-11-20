#!/bin/bash
# One Stage A task per array element. Submitted by orca/stageA_submit.sh, which
# sets TASKFILE (lines of "<size> <seed> <task>") and the GPU type.
#SBATCH --job-name=lcm-stageA
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=orca/logs/%x-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment.py ] || [ ! -f orca/env.sh ]; then
    echo "!! Submit from the repo root: cd ~/large-complexity-model"; exit 1
fi
source orca/env.sh || exit 1
source orca/stageA_settings.sh

line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKFILE")"
read -r size seed task <<< "$line"
if [ -z "${task:-}" ]; then
    echo "!! No task on line $SLURM_ARRAY_TASK_ID of $TASKFILE"; exit 1
fi
out="$STAGEA_ROOT/${size}_seed${seed}"
model_args="$(stageA_model_args "$size")"

echo "== Task $SLURM_ARRAY_JOB_ID[$SLURM_ARRAY_TASK_ID] on $(hostname) at $(date): $size seed $seed $task"
nvidia-smi --query-gpu=name --format=csv,noheader
git log --oneline -1 || echo "(git version unavailable)"
# shellcheck disable=SC2086
.venv/bin/python -u experiment.py --only "$task" --seed "$seed" --out "$out" $model_args "${STAGEA_COMMON[@]}"
echo "== Done at $(date)"
