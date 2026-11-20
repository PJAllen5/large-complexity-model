# Shared environment for ORCA scripts (sourced, not run).
# Makes the `module` command available in non-interactive shells, then loads
# the Python version this project uses on ORCA.

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi

LCM_PYTHON_MODULE="${LCM_PYTHON_MODULE:-python/3.12.12-gcc-13.4.0}"
if ! module load "$LCM_PYTHON_MODULE" 2>/dev/null; then
    echo "!! Could not load module $LCM_PYTHON_MODULE."
    echo "!! Run 'module avail python', then e.g.: export LCM_PYTHON_MODULE=python/<version>"
    return 1 2>/dev/null || exit 1
fi

# let scripts in orca/ import the lcm package (every script cd's to the repo root first)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
