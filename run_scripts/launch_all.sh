#!/usr/bin/env bash
# Train selected real-robot tasks sequentially, stopping on the first failure.
# DATASET_ROOT and CKPT_ROOT_BASE are required by jax_train_redflow.sh.
# Example: TASKS="folding_clothes table_cleaning" bash run_scripts/launch_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

read -r -a tasks <<< "${TASKS:-folding_clothes table_cleaning table_sweeping}"
log_dir=${LOG_DIR:-./logs}
mkdir -p "$log_dir"
for task in "${tasks[@]}"; do
    echo "Training $task; log: $log_dir/train_${task}_redflow.log"
    TASK="$task" bash run_scripts/jax_train_redflow.sh "$@" \
        2>&1 | tee "$log_dir/train_${task}_redflow.log"
done
printf 'Training finished. Checkpoints: %s/pi0_real_redflow/\n' "${CHECKPOINT_BASE_DIR:-./checkpoints}"
