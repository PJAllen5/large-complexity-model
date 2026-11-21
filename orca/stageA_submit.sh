#!/bin/bash
# Submit Stage A of the scale-up to Slurm. Run on the ORCA login node, from the
# repo root (it only submits jobs; nothing heavy runs here):
#
#     bash orca/stageA_submit.sh --check       # overfitting check: 4 specialist runs
#     bash orca/stageA_submit.sh               # full Stage A
#     bash orca/stageA_submit.sh --joint-long  # joint model only, 8x the steps (160k), all sizes and seeds,
#                                              # reusing the Stage A data; compare with the 20k-step specialists
#     bash orca/stageA_submit.sh --specialists-long  # the 8 specialists at 160k steps, 1.08M and 8.5M, all seeds,
#                                              # same data; compare with the 160k-step joint models
#
# What it submits:
#   1. a CPU job that simulates the shared data once (skipped if it exists)
#   2. a GPU job array on A30s  (1.08M and 8.5M models), at most 5 at once
#   3. a GPU job array on L40Ss (28M models), at most 3 at once
#   4. (full run only) a CPU job that writes each folder's report when all tasks succeed
# Tasks whose results already exist are left out, so rerunning this script after
# a failure only resubmits unfinished work. The --check runs are seed-0
# specialists in the same folders, so the full run reuses them.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-full}"
A30_MAX=5; L40S_MAX=3                                # at most 8 GPUs at once
USER_TASK_TIME="${TASK_TIME:-}"                      # an exported TASK_TIME overrides the per-mode limit
TASK_TIME=02:00:00
case "$MODE" in
    --check) MODE=check ;;
    --joint-long)
        MODE=joint-long
        # Each system gets as much training as a specialist: 8 systems x 20k steps.
        export STAGEA_ROOT="${STAGEA_ROOT:-$HOME/lcm-runs/stageA_joint160k}"
        export STAGEA_STEPS="${STAGEA_STEPS:-160000}"
        export STAGEA_DATA="${STAGEA_DATA:-$HOME/lcm-runs/stageA_n20k/data}"
        TASK_TIME=03:30:00 ;;                 # ~70-80 min expected; short partition allows 4 h
    --specialists-long)
        MODE=specialists-long
        # The fair comparison for the 160k-step joint models: specialists with the same 160k steps.
        export STAGEA_ROOT="${STAGEA_ROOT:-$HOME/lcm-runs/stageA_spec160k}"
        export STAGEA_STEPS="${STAGEA_STEPS:-160000}"
        export STAGEA_DATA="${STAGEA_DATA:-$HOME/lcm-runs/stageA_n20k/data}"
        TASK_TIME=03:00:00 ;;                 # set from the joint-long run times (sacct) before submitting
    full) ;;
    *) echo "usage: bash orca/stageA_submit.sh [--check | --joint-long | --specialists-long]"; exit 1 ;;
esac
TASK_TIME="${USER_TASK_TIME:-$TASK_TIME}"
source orca/stageA_settings.sh
mkdir -p "$STAGEA_ROOT" orca/logs

if [ "$MODE" = check ]; then
    A30_SIZES=(8.5M); L40S_SIZES=(28M); SEEDS=(0); TASKS=(specialist_epidemic specialist_lorenz96)
elif [ "$MODE" = joint-long ]; then
    A30_SIZES=(1.08M 8.5M); L40S_SIZES=(28M); SEEDS=("${STAGEA_SEEDS[@]}"); TASKS=(joint)
elif [ "$MODE" = specialists-long ]; then
    # 28M left out for now (it tied with 8.5M). 8.5M stays on A30s like its joint model;
    # the quick 1.08M runs use the L40Ss.
    A30_SIZES=(8.5M); L40S_SIZES=(1.08M); SEEDS=("${STAGEA_SEEDS[@]}")
    TASKS=(specialist_predator_prey specialist_boids specialist_kuramoto specialist_lorenz96
           specialist_nbody specialist_gene_reg specialist_neurons specialist_epidemic)
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
echo "Unfinished tasks: $n_a30 on A30, $n_l40s on L40S (results in $STAGEA_ROOT, $STAGEA_STEPS steps)"
if [ "$n_a30" = 0 ] && [ "$n_l40s" = 0 ] && [ "$MODE" != full ]; then
    echo "Nothing to run."; exit 0
fi

prep=$(sbatch --parsable orca/stageA_prep.sbatch)
echo "Data job: $prep"

deps=""
if [ "$n_a30" -gt 0 ]; then
    a30=$(sbatch --parsable --dependency="afterok:$prep" --array="1-${n_a30}%${A30_MAX}" --gres=gpu:a30:1 --time="$TASK_TIME" \
          --export=ALL,TASKFILE="$STAGEA_ROOT/tasks_${MODE}_a30.txt" orca/stageA_task.sbatch)
    echo "A30 array: $a30 ($n_a30 tasks)"; deps="$deps:$a30"
fi
if [ "$n_l40s" -gt 0 ]; then
    l40s=$(sbatch --parsable --dependency="afterok:$prep" --array="1-${n_l40s}%${L40S_MAX}" --gres=gpu:l40s:1 --time="$TASK_TIME" \
           --export=ALL,TASKFILE="$STAGEA_ROOT/tasks_${MODE}_l40s.txt" orca/stageA_task.sbatch)
    echo "L40S array: $l40s ($n_l40s tasks)"; deps="$deps:$l40s"
fi

if [ "$MODE" = full ]; then
    report=$(sbatch --parsable ${deps:+--dependency=afterok$deps} orca/stageA_reports.sbatch)
    echo "Report job: $report (runs after every task succeeds)"
fi
echo "Watch with: squeue -u \$USER"
