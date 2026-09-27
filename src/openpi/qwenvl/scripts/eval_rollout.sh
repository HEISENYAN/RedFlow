#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

mode="pi06"  # pi06 | pair
gpu="${GPU:-3}"
task_name="blocks_ranking_rgb"
rollout_root="${ROLLOUT_ROOT:?Set ROLLOUT_ROOT to your local directory}"
pi06_checkpoint_dir="${PI06_CHECKPOINT_DIR:?Set PI06_CHECKPOINT_DIR to your local directory}"
pair_checkpoint_dir="/path/to/pair/strict_checkpoint"

if [[ "${mode}" == "pi06" ]]; then
  checkpoint_dir="${pi06_checkpoint_dir}"
else
  checkpoint_dir="${pair_checkpoint_dir}"
fi

output_root="${ROOT_DIR}/qwenvl_runs/eval_rollout_${task_name}_${mode}"
final_dir="${ROOT_DIR}/qwenvl_runs/eval_rollout_${task_name}_${mode}_final"
max_episodes=10

args="
  --pipeline_mode episode \
  --model_type ${mode} \
  --source_mode hdf5 \
  --input_root ${rollout_root} \
  --checkpoint_dir ${checkpoint_dir} \
  --task_name ${task_name} \
  --output_root ${output_root} \
  --start_phase 0 \
  --end_phase 4 \
  --camera_names cam_high \
  --mode video \
  --fps 30 \
  --robot_type aloha \
  --max_episodes_per_task ${max_episodes} \
  --batch_size 8 \
  --stream_chunk_size 128 \
  --frame_interval 1 \
  --relative_interval 50 \
  --save_plots true \
  --attn_implementation flash_attention_2 \
  --bf16 true \
  --device cuda \
  --final_dir ${final_dir} \
  --resume true"

echo "[eval_rollout] mode=${mode}"
echo "[eval_rollout] rollout_root=${rollout_root}"
echo "[eval_rollout] checkpoint_dir=${checkpoint_dir}"
echo "[eval_rollout] gpu=${gpu}"
echo "[eval_rollout] task_name=${task_name}"
echo "[eval_rollout] max_episodes=${max_episodes}"
echo "[eval_rollout] output_root=${output_root}"
echo "[eval_rollout] final_dir=${final_dir}"
echo "[eval_rollout] resume=true"
echo "[eval_rollout] launching single-task pipeline"

CUDA_VISIBLE_DEVICES="${gpu}" \
  ./.venv/bin/python src/openpi/qwenvl/eval/run_pipeline.py ${args}
