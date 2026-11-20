# Shared environment for ORCA scripts (sourced, not run).
# Loads the Python version this project uses on ORCA, points Python at the
# repo, and caps CPU threads at what the Slurm job reserved.

# The calling scripts use `set -euo pipefail`, but ORCA's module system (Lmod)
# reads unset variables and runs commands that fail on purpose, which would
# abort the job. Relax those settings here and restore them at the end.
_lcm_saved_flags="$-"      # read directly: a $(...) copy would lose the -e setting
set +eu

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi

LCM_PYTHON_MODULE="${LCM_PYTHON_MODULE:-python/3.12.12-gcc-13.4.0}"
module load "$LCM_PYTHON_MODULE"          # errors stay visible in the log
_lcm_load_status=$?

# repo root = the folder above this file (falls back to the current folder)
_lcm_src="${BASH_SOURCE[0]:-}"
if [ -n "$_lcm_src" ]; then
    LCM_ROOT="$(cd "$(dirname "$_lcm_src")/.." && pwd)"
else
    LCM_ROOT="$PWD"
fi
[ -d "$LCM_ROOT/lcm" ] || echo "!! Warning: $LCM_ROOT doesn't look like the repo root (no lcm/ folder)"
export PYTHONPATH="$LCM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# A 64-core node would otherwise start ~64 threads inside an 8-core job.
_lcm_threads="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$_lcm_threads" MKL_NUM_THREADS="$_lcm_threads" OPENBLAS_NUM_THREADS="$_lcm_threads"

case "$_lcm_saved_flags" in *e*) set -e ;; esac
case "$_lcm_saved_flags" in *u*) set -u ;; esac
unset _lcm_saved_flags _lcm_threads _lcm_src

if [ "$_lcm_load_status" -ne 0 ]; then
    unset _lcm_load_status
    echo "!! Could not load module $LCM_PYTHON_MODULE (see the module error above)."
    echo "!! Run 'module avail python', then e.g.: export LCM_PYTHON_MODULE=python/<version>"
    return 1 2>/dev/null || exit 1
fi
unset _lcm_load_status
