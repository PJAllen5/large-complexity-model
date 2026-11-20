#!/bin/bash
# Stage A: simulate the shared data once (CPU only). Submitted by orca/stageA_submit.sh.
#SBATCH --job-name=lcm-prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=orca/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source orca/env.sh || exit 1
source orca/stageA_settings.sh
.venv/bin/python -u experiment.py --prepare-data-only --out "$STAGEA_ROOT/prep" "${STAGEA_COMMON[@]}"
