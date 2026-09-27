#!/bin/bash
# RedFlow training (KTO-Hinge + cluster pull) on the chunked real_rollout dataset.
#
# Per-task usage:
#   TASK=table_cleaning bash run_scripts/jax_train_redflow.sh
#
# Required env vars (set before running, or override on the command line):
#   DATASET_ROOT      Root of the per-task lerobot_chunked datasets
#   CKPT_ROOT_BASE    Root of the per-task SFT checkpoints (must contain
#                     <task>/params and <task>/assets/<assets_id>/norm_stats.json)
#   CHECKPOINT_BASE_DIR  Where to write training checkpoints (default ./checkpoints)
#   WANDB_API_KEY     Your wandb key (only needed if --wandb_enabled is set)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export DISABLE_FLASH_ATTN_3=1
export USE_TF=0
export TRANSFORMERS_VERBOSITY=error

# Configuration — auto-resolve ckpt + norm_stats from TASK
TASK=${TASK:-folding_clothes}
case "$TASK" in
  folding_clothes)  CKPT_TASK=folding_cloth;   ASSETS_ID=folding_cloth   ;;
  table_cleaning)   CKPT_TASK=cleaning_table;  ASSETS_ID=aloha_cleaning  ;;
  table_sweeping)   CKPT_TASK=sweeping_table;  ASSETS_ID=aloha_sweeping  ;;
  *) echo "Unknown TASK=$TASK"; exit 1 ;;
esac
DATASET_ROOT=${DATASET_ROOT:?Set DATASET_ROOT to the lerobot_chunked dataset root}
CKPT_ROOT_BASE=${CKPT_ROOT_BASE:?Set CKPT_ROOT_BASE to the per-task SFT checkpoint root}
CKPT_ROOT=${CKPT_ROOT_BASE}/${CKPT_TASK}
PI0_CKPT=${CKPT_ROOT}/params
NORM_STATS_PATH=${CKPT_ROOT}/assets/${ASSETS_ID}/norm_stats.json
exp_name=${EXP_NAME:-real_redflow_${TASK}}
# Local FS only — orbax+tensorstore use fcntl file locks that some network FS
# returns EBUSY on. Save locally; sync after training if needed.
checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-./checkpoints}
gpu_num=${GPU_NUM:-8}

for required_dir in "${DATASET_ROOT}/${TASK}/lerobot_chunked" "$PI0_CKPT"; do
    [[ -d "$required_dir" ]] || { echo "Missing directory: $required_dir" >&2; exit 1; }
done
[[ -f "$NORM_STATS_PATH" ]] || { echo "Missing norm stats: $NORM_STATS_PATH" >&2; exit 1; }

echo "=== TASK=$TASK ==="
echo "    DATA:        ${DATASET_ROOT}/${TASK}/lerobot_chunked"
echo "    PI0_CKPT:    $PI0_CKPT"
echo "    NORM_STATS:  $NORM_STATS_PATH"
echo "    CHECKPOINT:  ${checkpoint_base_dir}/pi0_real_redflow/${exp_name}"

"${PYTHON:-python}" scripts/train_redflow.py pi0_real_redflow \
    --exp_name "${exp_name}" \
    --checkpoint_base_dir "${checkpoint_base_dir}" \
    --fsdp_devices "${gpu_num}" \
    --num_epochs 30 \
    --save_every_n_epochs 1 \
    --max_to_keep 100 \
    --log_interval 100 \
    --batch_size 32 \
    --num_workers 2 \
    --no-wandb_enabled \
    --data.repo_id "${DATASET_ROOT}/${TASK}/lerobot_chunked" \
    --norm_stats_path "${NORM_STATS_PATH}" \
    --weight_loader.params_path "${PI0_CKPT}" \
    --lr_schedule.warmup_steps 1000 \
    --lr_schedule.peak_lr 2e-5 \
    --adv_temperature 3.0 \
    --kto_gamma 0.3 \
    --margin_scale 1.0 \
    --pull_lambda 0.3 \
    --no-is_debug \
    "$@"
