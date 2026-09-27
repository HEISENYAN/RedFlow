
# Environment setting
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HOME}/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${HOME}/.cache}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}
export TRANSFORMERS_VERBOSITY=error

exp_name=robotwin_50_clean_3_9
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-./checkpoints}

# Training setting
gpu_num=${GPU_NUM:-8}
export real_dir=${REAL_DATA_DIR:?Set REAL_DATA_DIR to the directory containing your real-rollout datasets}
REAL_TASKS=(
    "3_13/fold_cloth_perfect"
    "3_13/fold_cloth_retry"
    "3_15/fold_cloth_perfect"
    "3_15/fold_cloth_retry"
)

uv run scripts/compute_norm_stats.py \
    --config-name pi0_base_real \
