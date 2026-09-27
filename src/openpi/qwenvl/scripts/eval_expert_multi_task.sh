#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

mode="pi06"  # pi06 | pair
gpus=(1)  # GPU IDs to use

# expert_repo_root points to the parent of all task directories; tasks are discovered automatically.
expert_repo_root="${EXPERT_REPO_ROOT:?Set EXPERT_REPO_ROOT to your local directory}"
pi06_checkpoint_dir="${PI06_CHECKPOINT_DIR:?Set PI06_CHECKPOINT_DIR to your local directory}"
pair_checkpoint_dir="/path/to/pair/strict_checkpoint"

if [[ "${mode}" == "pi06" ]]; then
  checkpoint_dir="${pi06_checkpoint_dir}"
else
  checkpoint_dir="${pair_checkpoint_dir}"
fi

output_root="${ROOT_DIR}/qwenvl_runs/eval_expert_multi_gpu_${mode}"
max_episodes=10
num_workers_per_task=8  # Number of worker processes for parallel episode processing

# Check that expert_repo_root exists.
if [[ ! -d "${expert_repo_root}" ]]; then
  echo "Error: expert_repo_root does not exist: ${expert_repo_root}"
  exit 1
fi

echo "[eval_expert_multi_gpu] mode=${mode}"
echo "[eval_expert_multi_gpu] expert_repo_root=${expert_repo_root}"
echo "[eval_expert_multi_gpu] checkpoint_dir=${checkpoint_dir}"
echo "[eval_expert_multi_gpu] gpus=${gpus[*]}"
echo "[eval_expert_multi_gpu] max_episodes=${max_episodes}"
echo "[eval_expert_multi_gpu] num_workers_per_task=${num_workers_per_task}"
echo "[eval_expert_multi_gpu] output_root=${output_root}"
echo "[eval_expert_multi_gpu] Auto-discovering tasks..."
echo ""

# Use Python to discover all tasks (directories containing LeRobot datasets).
tasks=$(PYTHONPATH=src python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

expert_repo_root = Path('${expert_repo_root}').expanduser().resolve()
valid_tasks = []
for task in sorted(expert_repo_root.iterdir()):
    task_dir = task / 'aloha-agilex_clean_50'
    print(task_dir)
    if not task_dir.is_dir():
        continue
    # Check for a valid LeRobot dataset containing meta/info.json or a data directory.
    if (task_dir / 'meta' / 'info.json').exists() or (task_dir / 'data').exists():
        try:
            # Try loading the metadata to validate the dataset.
            metadata = LeRobotDatasetMetadata(str(task_dir))
            if hasattr(metadata, 'episodes') and metadata.episodes:
                valid_tasks.append(task_dir.name)
        except Exception:
            # Skip the directory if loading fails.
            pass
print(' '.join(valid_tasks))
")

if [[ -z "${tasks}" ]]; then
  echo "Error: No tasks found in ${expert_repo_root}"
  exit 1
fi

read -ra task_array <<< "${tasks}"

echo "[eval_expert_multi_gpu] Found ${#task_array[@]} tasks: ${tasks}"
echo "[eval_expert_multi_gpu] Scheduling tasks across GPUs..."
echo ""

PYTHONPATH=src python -m openpi.qwenvl.eval.run_scheduler \
  --tasks "${task_array[@]}" \
  --gpus "${gpus[@]}" \
  --pipeline_mode episode \
  --model_type "${mode}" \
  --source_mode lerobot \
  --input_root "${expert_repo_root}" \
  --checkpoint_dir "${checkpoint_dir}" \
  --output_root "${output_root}" \
  --start_phase 1 \
  --end_phase 3 \
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
  --resume true
