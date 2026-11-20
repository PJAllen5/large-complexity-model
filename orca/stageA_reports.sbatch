#!/bin/bash
# Stage A: write the report for every finished <size>_seed<N> folder.
# Submitted by orca/stageA_submit.sh to run after all tasks succeed; can also be
# submitted by hand: sbatch --gres=gpu:a30:1 orca/stageA_reports.sbatch
#SBATCH --job-name=lcm-reports
#SBATCH --partition=short
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=orca/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source orca/env.sh || exit 1
source orca/stageA_settings.sh
for dir in "$STAGEA_ROOT"/*_seed*/; do
    name="$(basename "$dir")"            # e.g. 8.5M_seed1
    size="${name%_seed*}"; seed="${name##*_seed}"
    echo "== Report for $name"
    # shellcheck disable=SC2046
    .venv/bin/python -u experiment.py --report-only --seed "$seed" --out "${dir%/}" \
        $(stageA_model_args "$size") "${STAGEA_COMMON[@]}"
done
echo "== Scaling report"
.venv/bin/python orca/scaling_report.py "$STAGEA_ROOT"
echo "== Done at $(date)"
