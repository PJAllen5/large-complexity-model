# Stage A settings, shared by the submit script and every task (sourced, not run).
# Decided 2026-11-20 (see PROJECT_LOG.md): bf16, 20k steps, batch 32, lr 3e-4,
# 256 validation episodes, few-shot 256 episodes x 2,000 steps, seeds 0-2.

STAGEA_ROOT="${STAGEA_ROOT:-$HOME/lcm-runs/stageA}"
STAGEA_SEEDS=(0 1 2)
STAGEA_COMMON=(--precision bf16 --steps 20000 --batch 32 --lr 3e-4 --n-val 256
               --few-shot-eps 256 --few-shot-steps 2000 --data-dir "$STAGEA_ROOT/data")

# size name -> width / layers / heads
stageA_model_args() {
    case "$1" in
        1.08M) echo "--d 128 --layers 4 --heads 4" ;;
        8.5M)  echo "--d 256 --layers 8 --heads 8" ;;
        28M)   echo "--d 384 --layers 12 --heads 8" ;;
        *) echo "unknown size $1" >&2; return 1 ;;
    esac
}

# result files a task produces (used to skip finished work)
stageA_task_outputs() {
    case "$1" in
        transfer_*) local n="${1#transfer_}"; echo "loo_$n finetune_$n scratch_$n" ;;
        *) echo "$1" ;;
    esac
}
