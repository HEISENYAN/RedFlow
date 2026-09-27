#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

mode="pi06"  # pi06 | pair
gpus=(0 1 2 3)  # GPU IDs to use

# rollout_root points to the parent of all task directories; tasks are discovered automatically.
rollout_root="${ROLLOUT_ROOT:?Set ROLLOUT_ROOT to your local directory}"
pi06_checkpoint_dir="${PI06_CHECKPOINT_DIR:?Set PI06_CHECKPOINT_DIR to your local directory}"
pair_checkpoint_dir="/path/to/pair/strict_checkpoint"

if [[ "${mode}" == "pi06" ]]; then
  checkpoint_dir="${pi06_checkpoint_dir}"
else
  checkpoint_dir="${pair_checkpoint_dir}"
fi

output_root="${ROOT_DIR}/qwenvl_runs/eval_rollout_multi_gpu_${mode}"
final_dir="${ROOT_DIR}/qwenvl_runs/eval_rollout_multi_gpu_${mode}_final"
max_episodes=128
num_workers_per_task=8  # Number of worker processes for parallel episode processing

# Check that rollout_root exists.
if [[ ! -d "${rollout_root}" ]]; then
  echo "Error: rollout_root does not exist: ${rollout_root}"
  exit 1
fi

echo "[eval_rollout_multi_gpu] mode=${mode}"
echo "[eval_rollout_multi_gpu] rollout_root=${rollout_root}"
echo "[eval_rollout_multi_gpu] checkpoint_dir=${checkpoint_dir}"
echo "[eval_rollout_multi_gpu] gpus=${gpus[*]}"
echo "[eval_rollout_multi_gpu] max_episodes=${max_episodes}"
echo "[eval_rollout_multi_gpu] num_workers_per_task=${num_workers_per_task}"
echo "[eval_rollout_multi_gpu] output_root=${output_root}"
echo "[eval_rollout_multi_gpu] final_dir=${final_dir}"
echo "[eval_rollout_multi_gpu] Auto-discovering tasks..."
echo ""

# Use Python to discover all tasks (directories containing .hdf5 files).
tasks=$(PYTHONPATH=src python -c "
from pathlib import Path
from openpi.qwenvl.utils.convert_hdf5_to_lerobot import scan_task_hdf5_dirs

rollout_root = Path('${rollout_root}').expanduser().resolve()
task_to_files = scan_task_hdf5_dirs(rollout_root)
tasks = sorted(task_to_files.keys())
print(' '.join(tasks))
")

if [[ -z "${tasks}" ]]; then
  echo "Error: No tasks found in ${rollout_root}"
  exit 1
fi

read -ra task_array <<< "${tasks}"

echo "[eval_rollout_multi_gpu] Found ${#task_array[@]} tasks: ${tasks}"
echo "[eval_rollout_multi_gpu] Scheduling tasks across GPUs..."
echo ""

PYTHONPATH=src python -m openpi.qwenvl.eval.run_scheduler \
  --tasks "${task_array[@]}" \
  --gpus "${gpus[@]}" \
  --pipeline_mode episode \
  --model_type "${mode}" \
  --source_mode hdf5 \
  --input_root "${rollout_root}" \
  --checkpoint_dir "${checkpoint_dir}" \
  --output_root "${output_root}" \
  --start_phase 0 \
  --end_phase 4 \
  --camera_names cam_high \
  --mode video \
  --fps 30 \
  --robot_type aloha \
  --max_episodes_per_task ${max_episodes} \
  --num_workers_per_task ${num_workers_per_task} \
  --batch_size 8 \
  --stream_chunk_size 128 \
  --frame_interval 1 \
  --relative_interval 50 \
  --save_plots true \
  --save_videos true \
  --video_fps 30 \
  --attn_implementation flash_attention_2 \
  --bf16 true \
  --final_dir ${final_dir} \
  --resume true
