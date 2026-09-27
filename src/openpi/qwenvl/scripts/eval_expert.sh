#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER:-user}/.cache}"

UV_PYTHON="${ROOT_DIR}/.venv/bin/python"

gpu="${GPU:-1}"
mode="pi06"  # pi06 | pair

task_name="blocks_ranking_rgb"
expert_repo_root="${EXPERT_REPO_ROOT:?Set EXPERT_REPO_ROOT to your local directory}"
pi06_checkpoint_dir="${PI06_CHECKPOINT_DIR:?Set PI06_CHECKPOINT_DIR to your local directory}"
pair_checkpoint_dir="/path/to/pair/strict_checkpoint"

if [[ "${mode}" == "pi06" ]]; then
  checkpoint_dir="${pi06_checkpoint_dir}"
else
  checkpoint_dir="${pair_checkpoint_dir}"
fi

output_root="${ROOT_DIR}/qwenvl_runs/eval_expert_${task_name}_${mode}_special"
max_episodes=10

args="
  --pipeline_mode episode \
  --model_type ${mode} \
  --source_mode lerobot \
  --input_root ${expert_repo_root} \
  --checkpoint_dir ${checkpoint_dir} \
  --task_name ${task_name} \
  --output_root ${output_root} \
  --start_phase 1 \
  --end_phase 3 \
  --camera_names cam_high \
  --max_episodes ${max_episodes} \
  --batch_size 8 \
  --stream_chunk_size 128 \
  --frame_interval 1 \
  --relative_interval 50 \
  --save_plots true \
  --attn_implementation flash_attention_2 \
  --bf16 true \
  --device cuda"

CUDA_VISIBLE_DEVICES="${gpu}" \
  "${UV_PYTHON}" src/openpi/qwenvl/eval/run_pipeline.py ${args}
