#!/bin/bash
# Submit Stage A of the scale-up to Slurm. Run on the ORCA login node, from the
# repo root (it only submits jobs; nothing heavy runs here):
#
#     bash orca/stageA_submit.sh --check    # overfitting check: 4 specialist runs
#     bash orca/stageA_submit.sh            # full Stage A
#
# What it submits:
#   1. a CPU job that simulates the shared data once (skipped if it exists)
#   2. a GPU job array on A30s  (1.08M and 8.5M models), at most 6 at once
#   3. a GPU job array on L40Ss (28M models), at most 2 at once
#   4. (full run only) a CPU job that writes each folder's report when all tasks succeed
# Tasks whose results already exist are left out, so rerunning this script after
# a failure only resubmits unfinished work. The --check runs are seed-0
# specialists in the same folders, so the full run reuses them.
set -euo pipefail
cd "$(dirname "$0")/.."
source orca/stageA_settings.sh

MODE="${1:-full}"
case "$MODE" in
    --check) MODE=check ;;
    full) ;;
    *) echo "usage: bash orca/stageA_submit.sh [--check]"; exit 1 ;;
esac
mkdir -p "$STAGEA_ROOT" orca/logs

if [ "$MODE" = check ]; then
    A30_SIZES=(8.5M); L40S_SIZES=(28M); SEEDS=(0); TASKS=(specialist_epidemic specialist_lorenz96)
else
    A30_SIZES=(1.08M 8.5M); L40S_SIZES=(28M); SEEDS=("${STAGEA_SEEDS[@]}")
    TASKS=(joint specialist_predator_prey specialist_boids specialist_kuramoto specialist_lorenz96
           specialist_nbody specialist_gene_reg specialist_neurons specialist_epidemic
           transfer_neurons transfer_epidemic transfer_kuramoto)
fi

write_tasks() {   # $1 = task file, remaining args = sizes; prints the number of unfinished tasks
    local file="$1"; shift
    : > "$file"
    for size in "$@"; do
        for seed in "${SEEDS[@]}"; do
            for task in "${TASKS[@]}"; do
                local done=1
                for tag in $(stageA_task_outputs "$task"); do
                    [ -f "$STAGEA_ROOT/${size}_seed${seed}/runs/$tag.json" ] || done=0
                done
                [ "$done" = 1 ] || echo "$size $seed $task" >> "$file"
            done
        done
    done
    wc -l < "$file" | tr -d ' '
}

n_a30=$(write_tasks "$STAGEA_ROOT/tasks_${MODE}_a30.txt" "${A30_SIZES[@]}")
n_l40s=$(write_tasks "$STAGEA_ROOT/tasks_${MODE}_l40s.txt" "${L40S_SIZES[@]}")
echo "Unfinished tasks: $n_a30 on A30, $n_l40s on L40S (results in $STAGEA_ROOT)"
if [ "$n_a30" = 0 ] && [ "$n_l40s" = 0 ] && [ "$MODE" = check ]; then
    echo "Nothing to run."; exit 0
fi

prep=$(sbatch --parsable orca/stageA_prep.sbatch)
echo "Data job: $prep"

deps=""
if [ "$n_a30" -gt 0 ]; then
    a30=$(sbatch --parsable --dependency="afterok:$prep" --array="1-${n_a30}%6" --gres=gpu:a30:1 \
          --export=ALL,TASKFILE="$STAGEA_ROOT/tasks_${MODE}_a30.txt" orca/stageA_task.sbatch)
    echo "A30 array: $a30 ($n_a30 tasks)"; deps="$deps:$a30"
fi
if [ "$n_l40s" -gt 0 ]; then
    l40s=$(sbatch --parsable --dependency="afterok:$prep" --array="1-${n_l40s}%2" --gres=gpu:l40s:1 \
           --export=ALL,TASKFILE="$STAGEA_ROOT/tasks_${MODE}_l40s.txt" orca/stageA_task.sbatch)
    echo "L40S array: $l40s ($n_l40s tasks)"; deps="$deps:$l40s"
fi

if [ "$MODE" = full ]; then
    report=$(sbatch --parsable ${deps:+--dependency=afterok$deps} orca/stageA_reports.sbatch)
    echo "Report job: $report (runs after every task succeeds)"
fi
echo "Watch with: squeue -u \$USER"
