#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=100
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

TORCHRUN_BIN="${ROOT_DIR}/.venv/bin/torchrun"
ENTRY_FILE="${ROOT_DIR}/src/openpi/qwenvl/train/train_qwen_pi06.py"

# Set WANDB_API_KEY in your environment before enabling online logging.
export WANDB_PROJECT="value-function"
export WANDB_ENTITY="heisen0928-the-hong-kong-polytechnic-university"

# Key settings
data_variant="expert"  # expert | rollout
dataset_root="/mnt/bn/ic-vlm/zhufangqi/code/datasets/robotwin_50_lerobot/"
dataset_leaf_name="aloha-agilex_randomized_500"
llm_path="/mnt/bn/ic-vlm/zhufangqi/code/checkpoints/qwen-vl-resize"
deepspeed_config="${ROOT_DIR}/src/openpi/qwenvl/scripts/zero3.json"
output_root="$/mnt/bn/ic-vlm/zhufangqi/code/checkpoints/value_function_pi06"
run_name="robotwin50_pi06_${data_variant}_train"

# Distributed training configuration
MASTER_ADDR="${MASTER_ADDR:-26.142.18.106}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-2}"
num_gpus=8

# Training hyperparameters
batch_size=16
grad_accum_steps=8
max_steps=30000
eval_interval=2500
save_interval=2500

dataset_paths=()
while IFS= read -r candidate; do
  dataset_paths+=("${candidate}")
done < <(find "${dataset_root}" -mindepth 2 -maxdepth 2 -type d -name "${dataset_leaf_name}" | sort)

if [[ ${#dataset_paths[@]} -eq 0 ]]; then
  echo "No datasets found under ${dataset_root} matching */${dataset_leaf_name}" >&2
  exit 1
fi

datasets="$(printf "%s," "${dataset_paths[@]}")"
datasets="${datasets%,}"
output_dir="${output_root}/${run_name}"

if [[ "${data_variant}" == "expert" ]]; then
  value_args="
    --value_source episode_progress \
    --value_tokenizer_min 0.0 \
    --value_tokenizer_max 1.0"
elif [[ "${data_variant}" == "rollout" ]]; then
  value_args="
    --value_source dataset \
    --value_column value \
    --value_tokenizer_min 0.0 \
    --value_tokenizer_max 1.0"
else
  echo "Unsupported data_variant: ${data_variant}" >&2
  exit 1
fi

echo "[train_pi06] data_variant=${data_variant}"
echo "[train_pi06] dataset_root=${dataset_root}"
echo "[train_pi06] dataset_count=${#dataset_paths[@]}"
printf '[train_pi06] dataset=%s\n' "${dataset_paths[@]}"
echo "[train_pi06] output_dir=${output_dir}"
echo "[train_pi06] num_gpus=${num_gpus}"
echo "[train_pi06] master_addr=${MASTER_ADDR}"
echo "[train_pi06] master_port=${MASTER_PORT}"
echo "[train_pi06] nnodes=${NNODES}"
echo "[train_pi06] deepspeed_config=${deepspeed_config}"
echo "[train_pi06] max_steps=${max_steps}"
echo "[train_pi06] eval_interval=${eval_interval}"
echo "[train_pi06] save_interval=${save_interval}"

args="
  --deepspeed ${deepspeed_config} \
  --model_name_or_path ${llm_path} \
  --dataset_use ${datasets} \
  --camera_names cam_high \
  --pair_mode false \
  --use_value_tokenizer true \
  ${value_args} \
  --tune_mm_vision false \
  --tune_mm_mlp true \
  --tune_mm_llm true \
  --bf16 true \
  --output_dir ${output_dir} \
  --max_steps ${max_steps} \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size ${batch_size} \
  --gradient_accumulation_steps ${grad_accum_steps} \
  --image_height 240 \
  --image_width 320 \
  --min_pixels 76800 \
  --max_pixels 76800 \
  --eval_strategy steps \
  --eval_steps ${eval_interval} \
  --save_strategy steps \
  --save_steps ${save_interval} \
  --save_total_limit 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --logging_strategy steps \
  --logging_steps 10 \
  --model_max_length 512 \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --save_format hf \
  --run_name ${run_name}"

"${TORCHRUN_BIN}" \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=26.142.18.106 \
  --master_port=29500 \
  "${ENTRY_FILE}" ${args}
