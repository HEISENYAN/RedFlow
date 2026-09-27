# RedFlow: Real-Robot Offline Training

This repository builds on [openpi](https://github.com/Physical-Intelligence/openpi) and implements RedFlow fine-tuning for π₀: a KTO-Hinge loss with cluster pull, using LoRA to update the vision-language model and action expert. The real-robot training entry point is [`run_scripts/jax_train_redflow.sh`](run_scripts/jax_train_redflow.sh), using the `pi0_real_redflow` configuration.

This guide covers **training from an already annotated chunked dataset and task-specific SFT weights**. The repository does not include real-robot datasets, task-specific SFT weights, or the complete pipeline for generating clusters, advantages, and pull targets. Reproducing the full experiment from raw recordings therefore requires additional resources. See the [data specification](docs/redflow_data.md) for the input contract. For upstream models, SFT, and other robot examples, see the [original openpi guide](README_UPSTREAM.md).

## 1. Installation

Run these commands from the repository root. Use Linux, Python 3.11, and an NVIDIA GPU environment compatible with the locked JAX CUDA 12 dependencies. The launcher defaults to 8 FSDP devices and a global batch size of 32. Measure memory requirements on your target hardware; upstream SFT memory estimates do not directly apply.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --python 3.11 --no-default-groups --group dev
source .venv/bin/activate
python -c 'import jax; print(jax.__version__); print(jax.devices())'
```

RedFlow training does not depend on the robot environments in `third_party`. If you cloned the repository with Git and want to use the upstream ALOHA/LIBERO environment examples, also run `git submodule update --init --recursive`.

Dependency versions are pinned in `uv.lock`. RedFlow does not require the optional QwenVL/DeepSpeed dependencies, so the command above installs only the main dependencies and the development group. Initial model use may require internet access to download public resources such as tokenizers.

## Published RedFlow Weights

The clothes-folding checkpoint is available at [Heisen0928/redflow_results — clothes_folding_redflow](https://huggingface.co/Heisen0928/redflow_results/tree/main/clothes_folding_redflow). These are RedFlow result weights; task-specific SFT initialization weights for reproducing training must be prepared separately.

Download the directory using the environment installed above:

```bash
python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Heisen0928/redflow_results",
    allow_patterns=["clothes_folding_redflow/*"],
    local_dir="checkpoints/published",
)
PYTHON
```

Inspect the downloaded files under `checkpoints/published/clothes_folding_redflow`. For inference, `--policy.dir` must point to the checkpoint directory that directly contains `params/` and `assets/`. If the download contains step-number subdirectories, select the appropriate one. Source checks do not verify loading these remote weights. The default `pi0_real_redflow` inference configuration expects `assets/real_rollout/norm_stats.json`. If the published weights use a different asset ID, use their matching configuration or explicitly pass their normalization statistics to `create_trained_policy`. Do not substitute statistics from another task.

## 2. Prepare Data and SFT Weights

`DATASET_ROOT` is the parent directory of all task datasets. `CKPT_ROOT_BASE` is the parent directory of all task SFT checkpoints. Use the layout below; dataset and checkpoint task names differ.

| TASK | Relative dataset path | Relative SFT path | Normalization statistics relative to SFT directory |
| --- | --- | --- | --- |
| `folding_clothes` (default) | `folding_clothes/lerobot_chunked` | `folding_cloth` | `assets/folding_cloth/norm_stats.json` |
| `table_cleaning` | `table_cleaning/lerobot_chunked` | `cleaning_table` | `assets/aloha_cleaning/norm_stats.json` |
| `table_sweeping` | `table_sweeping/lerobot_chunked` | `sweeping_table` | `assets/aloha_sweeping/norm_stats.json` |

Each SFT directory must also contain an Orbax-format `params/` directory. Use SFT weights compatible with the task's action representation, normalization statistics, and model architecture. Public π₀ base weights do not replace task-specific SFT weights when reproducing the same experiment. Do not replace the original SFT statistics with statistics recomputed on rollout data.

## 3. Start Training

```bash
export DATASET_ROOT=/absolute/path/to/real_rollout
export CKPT_ROOT_BASE=/absolute/path/to/sft_checkpoints
export CHECKPOINT_BASE_DIR=/absolute/path/to/local/checkpoints
TASK=folding_clothes GPU_NUM=8 bash run_scripts/jax_train_redflow.sh
```

Write checkpoints to a local disk that supports file locking. Some network filesystems do not satisfy Orbax/TensorStore locking requirements. `GPU_NUM` sets the FSDP shard count; use `CUDA_VISIBLE_DEVICES` to select GPUs. The global batch size must be divisible by the visible device count, and the FSDP count must be compatible with that count.

The defaults retain the original training settings: 30 epochs, one save per epoch, `max_to_keep=100` (also subject to the configuration's `keep_period` retention rule), batch size 32, 2 data workers, peak learning rate `2e-5`, 1000 warmup steps, `adv_temperature=3.0`, `kto_gamma=0.3`, `margin_scale=1.0`, and `pull_lambda=0.3`. See [`TrainConfig`](src/openpi/training/config.py) for parameters not explicitly set by the launcher.

Additional CLI arguments override the launcher's defaults. `EXP_NAME` sets the experiment name and defaults to `real_redflow_${TASK}`. W&B is disabled by default, and existing output directories are not overwritten automatically.

```bash
# Small run: still requires real data and SFT weights.
TASK=folding_clothes CUDA_VISIBLE_DEVICES=0 GPU_NUM=1 EXP_NAME=smoke \
  bash run_scripts/jax_train_redflow.sh --num_epochs 1 --batch_size 1 --num_workers 0

# Resume the same experiment; num_epochs is the total number of epochs.
bash run_scripts/jax_train_redflow.sh --resume

# Use --overwrite only when you intend to discard the existing experiment.
# Set WANDB_API_KEY or run wandb login before enabling W&B.
bash run_scripts/jax_train_redflow.sh --wandb_enabled
```

## 4. Inspect Outputs

Logs report the filtered `dataset_size`, `steps_per_epoch`, and `total_steps`, followed by training metrics. Rows with `cluster_label == -2` are excluded. Steps per epoch are `max(1, dataset_size // batch_size)`; ensure that the filtered dataset contains at least one complete batch.

Checkpoints use this layout:

```text
<CHECKPOINT_BASE_DIR>/pi0_real_redflow/<EXP_NAME>/<step>/
  params/
  assets/real_rollout/norm_stats.json
  ...
```

`<step>` is the training step, not the epoch number. Producing model outputs, finite losses, and saved checkpoints demonstrates that training runs. Real-robot performance still requires evaluation with consistent tasks, initial conditions, and success criteria.

Load a trained checkpoint with the inference server:

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_real_redflow \
  --policy.dir=/absolute/path/to/checkpoints/pi0_real_redflow/real_redflow_folding_clothes/STEP
```

The client must provide CHW-format RGB uint8 `images` (`cam_high`, `cam_left_wrist`, and `cam_right_wrist`), a 14-dimensional `state`, and a task `prompt`. These inference keys differ from the raw parquet column names. See the [remote inference guide](docs/remote_inference.md) for the client protocol.

## 5. Validation and Troubleshooting

```bash
bash -n run_scripts/jax_train_redflow.sh
python -c 'import openpi.training.config; import scripts.train_redflow'
pytest tests/test_redflow_losses.py tests/test_data_loader_chunked.py -q
# Optional: provide real data to enable the dataset integration test.
TEST_REAL_ROLLOUT_DIR="$DATASET_ROOT/folding_clothes/lerobot_chunked" \
  pytest tests/test_data_loader_chunked.py -q
```

- Missing RL fields: ordinary per-frame LeRobot datasets cannot be used directly with this entry point. See the data specification.
- Normalization dimension errors: check the 14-dimensional `actions` and `state` statistics and the target action coordinate convention.
- GPUs not visible: inspect `jax.devices()` before starting training.
- Out of memory: reduce batch size or adjust the FSDP device count. Record any configuration changes to distinguish the run from the default experiment.
- Output directory already exists: choose a new `EXP_NAME` or explicitly use `--resume`.

## Source Releases

Run `python scripts/prepare_public_release.py` to check the current source and create `dist/redflow-source.zip` without the old Git history. See the [release guide](docs/releasing.md) for publication instructions, third-party submodules, and validation scope.

## Code and Licenses

- [Training loop](scripts/train_redflow.py)
- [Model and loss](src/openpi/models/pi0_redflow.py)
- [Chunked data loader](src/openpi/training/data_loader_chunked.py)
- [Contributing](CONTRIBUTING.md)

The upstream [LICENSE](LICENSE) and [Gemma license](LICENSE_GEMMA.txt) are retained. Third-party code, models, and datasets remain subject to their respective providers' terms.
# RedFlow
