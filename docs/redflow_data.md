# RedFlow Data Specification

Training reads a local LeRobot v2.1 `lerobot_chunked/` directory containing `meta/info.json`, task and episode metadata, parquet data, and the images or videos referenced by that metadata. Copying only parquet files is insufficient. See [data_loader_chunked.py](../src/openpi/training/data_loader_chunked.py) for the implementation.

Each row represents one action chunk. The loader does not construct a chunk by reading the next 50 frames. The required fields are:

| Field | Per-row shape/type | Meaning |
| --- | --- | --- |
| `observation.images.cam_high` | RGB image | Main camera view |
| `observation.images.cam_left_wrist` | RGB image | Left wrist camera |
| `observation.images.cam_right_wrist` | RGB image | Right wrist camera |
| `observation.state` | float `[14]` | Dual-arm state: 6 joints and 1 gripper per arm |
| `action` | float `[700]` | Flattened 50 × 14 action chunk, in temporal order |
| `pull_target_action` | float `[700]` | Corresponding pull target action chunk |
| `cluster_advantage` | float scalar or `[1]` | Precomputed advantage |
| `pull_alpha` | float scalar or `[1]` | Pull term weight |
| `cluster_label` | int scalar or `[1]` | `-2` marks post-success chunks, which are filtered out |
| `is_success` | bool scalar or `[1]` | Success label |
| `progress_value` | float scalar or `[1]` | Progress value |
| `episode_index`, `frame_index`, `task_index` | int | LeRobot indices; task_index selects the task text |

All RL fields must be written to the dataset in advance. Training does not compute clusters, advantages, or target actions online. The repository's general format converters do not replace this preprocessing step, and the training code alone does not uniquely specify the algorithm that generated these fields.

## Action Coordinates and Normalization

`pi0_real_redflow` uses `adapt_to_pi=False` but inherits `use_delta_joint_actions=True`. For ordinary `action` values, the current joint state is subtracted first (6 dimensions per arm; grippers remain absolute), followed by normalization with the SFT `actions.mean/std` statistics.

**The current loader applies only mean/std normalization and zero-padding to `pull_target_action`; it does not convert joint targets to deltas.** This field must therefore already use the same delta-action coordinates described above and must not yet be normalized. If preprocessing stores absolute joint targets, convert them first. Otherwise, the two loss terms optimize targets in different coordinate systems. Confirm the actual data-generation convention before publishing experimental datasets.

Both action arrays are padded from 14 dimensions to the model's 32 dimensions, with a fixed horizon of 50. Changing robot degrees of freedom or the horizon requires coordinated changes to the model, loader, statistics, and preprocessing pipeline, not just the launch command.

The statistics file accepts openpi's `{"norm_stats": {"actions": {...}, "state": {...}}}` format, or the same dictionary without the outer `norm_stats` key. Each entry must include 14-dimensional `mean` and `std` arrays. The statistics must match the action coordinates used during SFT.

## Reproduction Scope

This repository provides training code that consumes the data described above and tests for the loss. Raw rollouts, the RL field generator, task-specific SFT weights, and the complete real-robot evaluation protocol are not included in this release. Without those inputs, you can inspect the code and interfaces but cannot claim reproduction of the original experimental results.
