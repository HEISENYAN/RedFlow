# `openpi.qwenvl`

`openpi.qwenvl` is the module containing the Qwen2.5-VL value function implementation in this repository.

Current constraints:

- Only `Qwen2.5-VL` is supported.
- Training, evaluation, and runtime inference share processor configuration recovery, Qwen chat message construction, RoPE, and batch collation.
- Strict validation is the default; no fallback behavior is provided.

## Directory Structure

```text
openpi/qwenvl/
├── data/
│   ├── __init__.py
│   ├── _lerobot_dataset.py
│   ├── _module_common.py
│   ├── data_loader_pair_map.py
│   ├── data_loader_pi06_map.py
│   ├── data_module.py
│   ├── data_processor.py
│   ├── data_processor_pair_map.py
│   ├── data_processor_pi06_map.py
│   └── rope2d.py
├── eval/
│   ├── __init__.py
│   ├── _artifacts.py
│   ├── _curve_ops.py
│   ├── phase1_eval_raw.py
│   ├── phase2_smooth.py
│   ├── phase3_segment.py
│   ├── phase4_rewrite.py
│   ├── run_pipeline.py
│   └── run_scheduler.py
├── modeling/
│   ├── _qwen_vl_shared.py
│   ├── pair_inference.py
│   └── pi06_inference.py
├── train/
│   ├── __init__.py
│   ├── argument.py
│   ├── pair_model.py
│   ├── train_qwen_pair.py
│   └── train_qwen_pi06.py
├── utils/
│   ├── convert_hdf5_to_lerobot.py
│   └── value_tokenizer.py
├── README.md
└── value_function.py
```

Dependencies are defined in the root `pyproject.toml` under the `qwenvl` group.

## Module Responsibilities

### `modeling/`

- [`modeling/_qwen_vl_shared.py`](modeling/_qwen_vl_shared.py) provides the shared implementation for training, evaluation, and runtime inference: recovering `DataArguments` from `run_config.json`, synchronizing processor pixel settings, constructing Qwen chat/generation messages, selecting Qwen2.5-VL RoPE, collating padded or flattened batches, and moving batches to the target device.
- [`modeling/pi06_inference.py`](modeling/pi06_inference.py) handles single-frame, three-view `pi06` inference. It validates checkpoints, restores the value tokenizer, builds pi06 prompts, assembles training and inference samples consistently, and decodes the final token logits into values.
- [`modeling/pair_inference.py`](modeling/pair_inference.py) compares current and historical frames in `pair` mode. It builds pair prompts, assembles training and inference samples consistently, and performs batched inference with delta outputs.

### `data/`

- [`data/_lerobot_dataset.py`](data/_lerobot_dataset.py) is the shared LeRobot reader. Training and evaluation use it for episode metadata, instructions, parquet paths, and frame triplets. Missing instructions raise an error instead of falling back to default text.
- [`data/data_loader_pi06_map.py`](data/data_loader_pi06_map.py) provides the `pi06` training dataset and requires LeRobot input. `value_source=dataset` requires the canonical `value` column. `value_source=episode_progress` explicitly selects `t/T`-style progress labels. Training targets map values from `[0, 1]` to `[-1, 0]`.
- [`data/data_loader_pair_map.py`](data/data_loader_pair_map.py) provides the `pair` training dataset, returning `current_images + history_images + delta_label`.
- [`data/data_processor_pi06_map.py`](data/data_processor_pi06_map.py) connects pi06 dataset samples to the shared modeling helpers.
- [`data/data_processor_pair_map.py`](data/data_processor_pair_map.py) connects pair dataset samples to the shared modeling helpers.
- [`data/data_module.py`](data/data_module.py) is the common training data-module entry point. `pair_mode=True` selects pair mode; otherwise it uses pi06.
- [`data/data_processor.py`](data/data_processor.py) contains minimal shared data utilities, primarily camera-name parsing.

### `train/`

- [`train/train_qwen_pi06.py`](train/train_qwen_pi06.py) is the pi06 training entry point. It uses shared processor recovery and training data modules instead of maintaining separate processor/collation logic.
- [`train/train_qwen_pair.py`](train/train_qwen_pair.py) is the pair training entry point. It also uses shared processor recovery and training data modules.
- [`train/pair_model.py`](train/pair_model.py) defines, exports, and loads the pair model. Only the Qwen2.5-VL backbone is supported.
- [`train/argument.py`](train/argument.py) defines `ModelArguments`, `DataArguments`, and `TrainingArguments`.

### `eval/`

The evaluation implementation is organized into phases 1–4:

- [`eval/phase1_eval_raw.py`](eval/phase1_eval_raw.py) consumes a LeRobot dataset and a strictly validated checkpoint directory, producing per-frame raw metrics: `absolute_value`, `relative_advantage`, `absolute_advantage`, metrics CSV/JSONL files, and a manifest.
- [`eval/phase2_smooth.py`](eval/phase2_smooth.py) consumes phase 1 outputs and produces `progress01_raw` and `progress01_smooth`.
- [`eval/phase3_segment.py`](eval/phase3_segment.py) consumes phase 2 outputs and produces peaks, valleys, pairs, and terminal anchors.
- [`eval/phase4_rewrite.py`](eval/phase4_rewrite.py) consumes phase 1–3 manifests and produces rewritten LeRobot parquet files or an exported dataset copy.
- [`eval/run_pipeline.py`](eval/run_pipeline.py) is the single-task entry point. With `source_mode=hdf5`, it first runs phase 0 conversion, then phases 1–4. With `source_mode=lerobot`, it starts directly at phase 1.
- [`eval/run_scheduler.py`](eval/run_scheduler.py) schedules tasks across GPUs. Each GPU handles a complete task pipeline; tasks are not split by episode across GPUs.
- [`eval/_curve_ops.py`](eval/_curve_ops.py) implements curve smoothing, peak/valley detection, and segment reconstruction.
- [`eval/_artifacts.py`](eval/_artifacts.py) reads and writes JSON, JSONL, CSV, and manifest files.

### `utils/`

- [`utils/convert_hdf5_to_lerobot.py`](utils/convert_hdf5_to_lerobot.py) implements phase 0: converting RoboTwin rollout HDF5 data to LeRobot.
- [`utils/value_tokenizer.py`](utils/value_tokenizer.py) converts between continuous values and discrete tokens.

### `value_function.py`

[`value_function.py`](value_function.py) provides runtime value inference for direct use by AWR or other training logic. Both pi06 and pair modes use the shared modeling implementation.

## Shared Conventions

### 1. No Fallbacks

- Missing instructions raise an error.
- Missing checkpoint files raise an error.
- A missing `run_config.json` raises an error.
- `value_source` must be explicit; `value_source=dataset` raises an error if the `value` column is missing.
- `source_mode` must be explicit.

### 2. Qwen2.5-VL Only

Training, evaluation, and runtime inference support only `Qwen2.5-VL`.

### 3. Consistency Across Training, Evaluation, and Runtime

All three paths share the same recovery of `DataArguments` from `run_config.json`, processor pixel settings, Qwen chat message construction, RoPE selection, and batch collation/padding. Image sizes, video parameters, and tokenizer settings are therefore not maintained independently by training and evaluation.

## Recommended Workflow

### Phase 0: HDF5 to LeRobot

```bash
# Run from the repository root
PYTHONPATH=src python -m openpi.qwenvl.utils.convert_hdf5_to_lerobot \
  --input_root /path/to/hdf5_root \
  --output_root /path/to/lerobot_root
```

### Run the Full Pipeline for One Task

LeRobot input:

```bash
# Run from the repository root
PYTHONPATH=src python -m openpi.qwenvl.eval.run_pipeline \
  --source_mode lerobot \
  --input_root /path/to/lerobot_repo \
  --checkpoint_dir /path/to/checkpoint_dir \
  --task_name some_task \
  --output_root /path/to/output_root
```

HDF5 input:

```bash
# Run from the repository root
PYTHONPATH=src python -m openpi.qwenvl.eval.run_pipeline \
  --source_mode hdf5 \
  --input_root /path/to/hdf5_task_root \
  --checkpoint_dir /path/to/checkpoint_dir \
  --task_name some_task \
  --output_root /path/to/output_root
```

### Schedule Tasks Across GPUs

```bash
# Run from the repository root
PYTHONPATH=src python -m openpi.qwenvl.eval.run_scheduler \
  --source_mode hdf5 \
  --input_root /path/to/task_root_parent \
  --checkpoint_dir /path/to/checkpoint_dir \
  --output_root /path/to/output_root \
  --tasks task_a task_b \
  --gpus 0 1
```

## Inputs and Checkpoints

Expert data should use LeRobot format; rollout input may use HDF5. Provide your own dataset directories and QwenVL checkpoints, which are not distributed with this repository. A checkpoint must contain the training `run_config.json`, processor configuration, and model weights.

This module is an optional value-function tool. It is not the complete cluster, advantage, and pull-target generator missing from the root RedFlow guide. Install the `qwenvl` dependency group before use.

Evaluation scripts accept input paths through `EXPERT_REPO_ROOT` or `ROLLOUT_ROOT`. The default pi06 mode takes its checkpoint path from `PI06_CHECKPOINT_DIR`. You can also use the Python CLI examples above directly.

## Code Navigation

To inspect training/evaluation consistency, start with these files:

- Shared implementation: [`modeling/_qwen_vl_shared.py`](modeling/_qwen_vl_shared.py).
- Pi06: [`modeling/pi06_inference.py`](modeling/pi06_inference.py), [`train/train_qwen_pi06.py`](train/train_qwen_pi06.py), and [`data/data_processor_pi06_map.py`](data/data_processor_pi06_map.py).
- Pair: [`modeling/pair_inference.py`](modeling/pair_inference.py), [`train/train_qwen_pair.py`](train/train_qwen_pair.py), and [`data/data_processor_pair_map.py`](data/data_processor_pair_map.py).
- Evaluation phases: [`eval/run_pipeline.py`](eval/run_pipeline.py), [`eval/phase1_eval_raw.py`](eval/phase1_eval_raw.py), [`eval/phase2_smooth.py`](eval/phase2_smooth.py), [`eval/phase3_segment.py`](eval/phase3_segment.py), and [`eval/phase4_rewrite.py`](eval/phase4_rewrite.py).

For data loading and prompt sources, see [`data/_lerobot_dataset.py`](data/_lerobot_dataset.py), [`data/data_loader_pi06_map.py`](data/data_loader_pi06_map.py), and [`data/data_loader_pair_map.py`](data/data_loader_pair_map.py).
