
# Environment setting (override these before running, or set in your shell)
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HOME}/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${HOME}/.cache}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${HOME}/.cache/openpi}
export TRANSFORMERS_VERBOSITY=error

exp_name=aloha_sweeping
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-./checkpoints}

# Training setting
gpu_num=${GPU_NUM:-8}

real_dir=${REAL_DATA_DIR:?Set REAL_DATA_DIR to the directory containing your real-rollout datasets}
REAL_TASKS=(
    "3_13/fold_cloth_perfect"
    "3_15/fold_cloth_perfect"
    "3_17/fold_cloth_perfect"
)

uv run scripts/train.py pi0_base_real \
    --exp_name ${exp_name} \
    --checkpoint_base_dir ${checkpoint_base_dir} \
    --fsdp_devices ${gpu_num} \
    --num_train_steps 500000 \
    --data.repo_id ${REAL_TASKS[@]/#/$real_dir}  \
    --wandb_enabled \
    --log_interval 100 \
    --batch_size 128 \
    --save_interval 10000 \
    --no-is_debug \
    --assets_base_dir ./pre-defined-assets/ \
    --resume
