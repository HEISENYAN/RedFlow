from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
import types
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Sequence

import h5py
import numpy as np
from lerobot.datasets.video_utils import encode_video_frames

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


@dataclass(frozen=True)
class HDF5FeatureSchema:
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    image_shape: tuple[int, int, int]
    image_shape_order: Literal["HWC", "CHW"]  # HWC: (height, width, channels), CHW: (channels, height, width)


@dataclass
class EpisodeData:
    """Preprocessed episode data, without a LeRobotDataset object."""

    hdf5_path: Path
    episode_pos: int
    effective_length: int
    state: np.ndarray  # (effective_length, state_dim)
    action: np.ndarray  # (effective_length, action_dim)
    images: dict[str, np.ndarray]  # {camera_name: (effective_length, H, W, C)}
    rewards: np.ndarray
    timesteps: np.ndarray
    episode_success: np.ndarray
    prompts: list[str]
    task_text: str
    seed_value: int
    task_id_value: int
    resolved_result: int
    # Record a warning for truncation due to mismatched lengths so the main process can summarize it.
    length_mismatch_warning: dict[str, Any] | None = None


@dataclass
class SkippedEpisode:
    """An episode that must be skipped or could not be processed."""

    hdf5_path: Path
    episode_pos: int
    reason: str
    details: dict[str, Any]


def _load_lerobot_dataset_cls():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: list[Any] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "dataset"


def _feature_names(prefix: str, shape: tuple[int, ...]) -> list[str] | None:
    if len(shape) != 1:
        return None
    return [f"{prefix}_{index}" for index in range(int(shape[0]))]


def _to_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _detect_image_shape_order(shape: tuple[int, int, int]) -> Literal["HWC", "CHW"]:
    """
    Detect the order of image dimensions in HDF5.
    Common formats:
    - HWC: (height, width, channels) - e.g., (480, 640, 3)
    - CHW: (channels, height, width) - e.g., (3, 480, 640)
    
    Detection heuristic:
    - If last dim is 1 or 3 (typical channel values), likely HWC
    - If first dim is 1 or 3, likely CHW
    - If both are small, prefer HWC (more common in image processing)
    """
    dim0, dim1, dim2 = shape
    # Channels are typically 1 (grayscale) or 3 (RGB)
    is_channel_dim = lambda d: d in (1, 3, 4)
    
    if is_channel_dim(dim2) and not is_channel_dim(dim0):
        # Last dimension is channels -> HWC format
        return "HWC"
    elif is_channel_dim(dim0) and not is_channel_dim(dim2):
        # First dimension is channels -> CHW format
        return "CHW"
    else:
        # Ambiguous case: assume HWC (more common for image storage)
        # If both dim0 and dim2 could be channels, check which is more likely
        # Channels are usually smaller than spatial dimensions
        if dim0 < dim1 and dim0 < dim2:
            return "CHW"
        else:
            return "HWC"


def _normalize_image_shape(shape: tuple[int, int, int], order: Literal["HWC", "CHW"]) -> tuple[int, int, int]:
    """
    Convert image shape to standard (height, width, channels) format.
    """
    if order == "HWC":
        return shape  # Already in correct format
    elif order == "CHW":
        channels, height, width = shape
        return (height, width, channels)
    else:
        raise ValueError(f"Unsupported image shape order: {order}")


def _convert_image_array_to_hwc(
    image_array: np.ndarray, 
    original_order: Literal["HWC", "CHW"]
) -> np.ndarray:
    """
    Convert image array from original order to (height, width, channels) format.
    Input array shape: (num_frames, ...) where ... is either (H, W, C) or (C, H, W)
    Output array shape: (num_frames, H, W, C)
    """
    if original_order == "HWC":
        return image_array  # Already in correct format
    elif original_order == "CHW":
        # Convert from (num_frames, C, H, W) to (num_frames, H, W, C)
        return np.transpose(image_array, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported image shape order: {original_order}")


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _to_scalar_int_attr(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_bool_attr(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "success"}:
            return True
        if lowered in {"0", "false", "no", "failure", "fail"}:
            return False
    return bool(value)


def _read_optional_array(handle: h5py.File, key: str, dtype: np.dtype | str | None = None) -> np.ndarray | None:
    if key not in handle:
        return None
    array = np.asarray(handle[key])
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _read_prompts(handle: h5py.File, length: int) -> list[str]:
    if "observation/prompt" not in handle:
        return [""] * int(length)
    raw_prompts = handle["observation/prompt"][:].tolist()
    prompts = [_decode_string(item) for item in raw_prompts]
    if len(prompts) < length:
        prompts.extend([""] * (length - len(prompts)))
    return prompts[:length]


def _dataset_length(array: np.ndarray | None) -> int:
    if array is None or array.ndim == 0:
        return 0
    return int(array.shape[0])


def _all_episode_lengths(handle: h5py.File) -> dict[str, int]:
    lengths = {
        "observation.state": _dataset_length(_read_optional_array(handle, "observation/state")),
        "action": _dataset_length(_read_optional_array(handle, "action")),
        "reward": _dataset_length(_read_optional_array(handle, "reward")),
        "timestep": _dataset_length(_read_optional_array(handle, "timestep")),
        "episode_success": _dataset_length(_read_optional_array(handle, "episode_success")),
        "episode_length": _dataset_length(_read_optional_array(handle, "episode_length")),
        "observation.prompt": _dataset_length(_read_optional_array(handle, "observation/prompt")),
    }
    for camera_name in CAMERA_NAMES:
        lengths[f"observation.images.{camera_name}"] = _dataset_length(
            _read_optional_array(handle, f"observation/images/{camera_name}")
        )
    return lengths


def scan_task_hdf5_dirs(input_root: str | Path) -> dict[str, list[Path]]:
    input_root = Path(input_root).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input rollout root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input rollout root must be a directory: {input_root}")

    task_to_files: dict[str, list[Path]] = {}
    for task_dir in sorted((path for path in input_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        hdf5_files = sorted(task_dir.glob("*.hdf5"), key=_natural_sort_key)
        if hdf5_files:
            task_to_files[task_dir.name] = hdf5_files
    return task_to_files


def infer_hdf5_feature_schema(hdf5_paths: Sequence[str | Path]) -> HDF5FeatureSchema:
    if not hdf5_paths:
        raise ValueError("No HDF5 files were provided for schema inference.")

    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as handle:
            state = _read_optional_array(handle, "observation/state")
            action = _read_optional_array(handle, "action")
            if state is None or action is None or state.ndim < 2 or action.ndim < 2:
                continue
            if state.shape[0] <= 0 or action.shape[0] <= 0:
                continue

            image_shape = None
            image_shape_order = None
            for camera_name in CAMERA_NAMES:
                camera_array = _read_optional_array(handle, f"observation/images/{camera_name}")
                if camera_array is None or camera_array.ndim != 4 or camera_array.shape[0] <= 0:
                    image_shape = None
                    break
                camera_shape = _to_tuple(camera_array.shape[1:])
                if len(camera_shape) != 3:
                    image_shape = None
                    break
                detected_order = _detect_image_shape_order(camera_shape)
                if image_shape is None:
                    image_shape = camera_shape
                    image_shape_order = detected_order
                elif camera_shape != image_shape:
                    raise ValueError(
                        f"Inconsistent camera shapes in {hdf5_path}: "
                        f"expected {image_shape}, got {camera_shape} for {camera_name}"
                    )
                elif detected_order != image_shape_order:
                    raise ValueError(
                        f"Inconsistent image shape order in {hdf5_path}: "
                        f"expected {image_shape_order}, got {detected_order} for {camera_name}"
                    )
            if image_shape is None or image_shape_order is None:
                continue

            # Normalize image shape to (height, width, channels) for consistent output
            normalized_image_shape = _normalize_image_shape(image_shape, image_shape_order)

            return HDF5FeatureSchema(
                state_shape=_to_tuple(state.shape[1:]),
                action_shape=_to_tuple(action.shape[1:]),
                image_shape=normalized_image_shape,
                image_shape_order=image_shape_order,
            )

    raise ValueError("Could not infer a non-empty Robotwin HDF5 schema from the provided files.")


def _build_lerobot_features(
    schema: HDF5FeatureSchema,
    mode: Literal["video", "image"],
    fps: int,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": schema.state_shape,
            "names": {"motors": MOTORS} if schema.state_shape == (len(MOTORS),) else _feature_names("state", schema.state_shape),
            "fps": int(fps),
        },
        "action": {
            "dtype": "float32",
            "shape": schema.action_shape,
            "names": {"motors": MOTORS} if schema.action_shape == (len(MOTORS),) else _feature_names("action", schema.action_shape),
            "fps": int(fps),
        },
        "reward": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["reward"],
            "fps": int(fps),
        },
        "result": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["result"],
            "fps": int(fps),
        },
        "episode_success": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["episode_success"],
            "fps": int(fps),
        },
        "episode_length": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["episode_length"],
            "fps": int(fps),
        },
        "timestep": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["timestep"],
            "fps": int(fps),
        },
        "task_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["task_id"],
            "fps": int(fps),
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["seed"],
            "fps": int(fps),
        },
        "prompt": {
            "dtype": "string",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
        "frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
        "episode_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
        "index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
        "task_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
            "fps": int(fps),
        },
    }
    for camera_name in CAMERA_NAMES:
        # schema.image_shape is already normalized to (height, width, channels)
        height, width, channels = schema.image_shape
        features[f"observation.images.{camera_name}"] = {
            "dtype": mode,
            "shape": (height, width, channels),
            "names": ["height", "width", "rgb"],
            "fps": int(fps),
        }
        if mode == "video":
            # Align image-related meta with Robotwin v3.0 dataset convention where possible.
            features[f"observation.images.{camera_name}"]["info"] = {
                "video.fps": float(fps),
                "video.height": int(height),
                "video.width": int(width),
                "video.channels": int(channels),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
    return features


def _resolve_effective_task_text(
    handle: h5py.File,
    prompts: Sequence[str],
    task_name: str,
) -> str:
    instruction = str(handle.attrs.get("instruction", "")).strip()
    if instruction:
        return instruction
    for prompt in prompts:
        if prompt.strip():
            return prompt.strip()
    return task_name


def _load_episode_worker(
    hdf5_path: Path,
    episode_pos: int,
    task_name: str,
    schema: HDF5FeatureSchema,
    fps: int,
) -> EpisodeData:
    """
    Worker function that reads and preprocesses HDF5 data without writing to the dataset.
    Runs in a separate process and is safe to execute in parallel.
    Raises exceptions on errors instead of returning SkippedEpisode.
    """
    with h5py.File(hdf5_path, "r") as handle:
        # Reuse the existing length checks.
        lengths = _all_episode_lengths(handle)
        positive_lengths = [length for length in lengths.values() if length > 0]
        declared_episode_length = _to_scalar_int_attr(handle.attrs.get("episode_length"), 0)
        
        if declared_episode_length == 0 or not positive_lengths:
            raise ValueError(
                f"Episode {episode_pos} ({hdf5_path}) is empty: "
                f"declared_episode_length={declared_episode_length}, lengths={lengths}"
            )
        
        effective_length = min(positive_lengths)
        if effective_length <= 0:
            raise ValueError(
                f"Episode {episode_pos} ({hdf5_path}) has non-positive effective length: "
                f"effective_length={effective_length}, declared_episode_length={declared_episode_length}, lengths={lengths}"
            )
        
        # Check for warnings about mismatched lengths.
        unique_lengths = {length for length in positive_lengths}
        length_mismatch_warning = None
        if len(unique_lengths) > 1 or declared_episode_length != effective_length:
            length_mismatch_warning = {
                "file": str(hdf5_path),
                "warning": "length_mismatch_truncated",
                "declared_episode_length": declared_episode_length,
                "effective_length": effective_length,
                "lengths": lengths,
            }
        
        # Read data using the existing logic.
        state = np.asarray(handle["observation/state"][:effective_length], dtype=np.float32)
        action = np.asarray(handle["action"][:effective_length], dtype=np.float32)
        images = {}
        for camera_name in CAMERA_NAMES:
            raw_image_array = np.asarray(
                handle[f"observation/images/{camera_name}"][:effective_length], dtype=np.uint8
            )
            images[camera_name] = _convert_image_array_to_hwc(raw_image_array, schema.image_shape_order)
        
        # Validate shapes.
        _validate_episode_shapes(hdf5_path, state, action, images, schema)
        
        # Read the remaining fields.
        rewards_array = _read_optional_array(handle, "reward", np.int64)
        timesteps_array = _read_optional_array(handle, "timestep", np.int64)
        episode_success_array = _read_optional_array(handle, "episode_success", np.int64)
        prompts = _read_prompts(handle, effective_length)
        
        # Apply defaults using the existing logic.
        if rewards_array is None:
            rewards_array = np.zeros((effective_length,), dtype=np.int64)
        else:
            rewards_array = rewards_array[:effective_length]
        
        if timesteps_array is None:
            timesteps_array = np.arange(effective_length, dtype=np.int64)
        else:
            timesteps_array = timesteps_array[:effective_length]
        
        resolved_result = int(_to_bool_attr(handle.attrs.get("result"), default=False))
        if episode_success_array is None:
            episode_success_array = np.full((effective_length,), resolved_result, dtype=np.int64)
        else:
            episode_success_array = episode_success_array[:effective_length]
        
        task_text = _resolve_effective_task_text(handle, prompts, task_name)
        seed_value = _to_scalar_int_attr(handle.attrs.get("seed"), -1)
        task_id_value = _to_scalar_int_attr(handle.attrs.get("task_id"), -1)
        
        return EpisodeData(
            hdf5_path=hdf5_path,
            episode_pos=episode_pos,
            effective_length=effective_length,
            state=state,
            action=action,
            images=images,
            rewards=rewards_array,
            timesteps=timesteps_array,
            episode_success=episode_success_array,
            prompts=prompts,
            task_text=task_text,
            seed_value=seed_value,
            task_id_value=task_id_value,
            resolved_result=resolved_result,
            length_mismatch_warning=length_mismatch_warning,
        )


def _validate_episode_shapes(
    hdf5_path: Path,
    state: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
    schema: HDF5FeatureSchema,
) -> None:
    if _to_tuple(state.shape[1:]) != schema.state_shape:
        raise ValueError(
            f"State shape mismatch for {hdf5_path}: expected {schema.state_shape}, got {_to_tuple(state.shape[1:])}"
        )
    if _to_tuple(action.shape[1:]) != schema.action_shape:
        raise ValueError(
            f"Action shape mismatch for {hdf5_path}: expected {schema.action_shape}, got {_to_tuple(action.shape[1:])}"
        )
    for camera_name, image_array in images.items():
        image_shape = _to_tuple(image_array.shape[1:])
        if image_shape != schema.image_shape:
            raise ValueError(
                f"Image shape mismatch for {hdf5_path}: expected {schema.image_shape}, "
                f"got {image_shape} for {camera_name}"
            )


def _finalize_dataset(dataset: Any) -> None:
    if hasattr(dataset, "_close_writer"):
        dataset._close_writer()
    metadata = getattr(dataset, "meta", None)
    if metadata is not None and hasattr(metadata, "_close_writer"):
        metadata._close_writer()


def _patch_video_tempdir(dataset: Any, temp_root: str | Path = "/tmp") -> None:
    temp_root = Path(temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        temp_path = Path(tempfile.mkdtemp(dir=temp_root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_video_frames(img_dir, temp_path, self.fps, vcodec="h264", overwrite=True)
        shutil.rmtree(img_dir)
        return temp_path

    dataset._encode_temporary_episode_video = types.MethodType(_encode_temporary_episode_video, dataset)


def _patch_save_episode_data(dataset: Any) -> None:
    def _save_episode_data(self, episode_buffer: dict) -> dict[str, Any]:
        import gc

        import datasets
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import embed_images
        from lerobot.datasets.lerobot_dataset import get_hf_dataset_size_in_mb
        from lerobot.datasets.lerobot_dataset import get_parquet_file_size_in_mb
        from lerobot.datasets.lerobot_dataset import get_parquet_num_frames
        from lerobot.datasets.lerobot_dataset import to_parquet_with_hf_images
        from lerobot.datasets.utils import update_chunk_file_indices

        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        ep_num_frames = len(ep_dataset)
        df = pd.DataFrame(ep_dataset)

        if self.meta.episodes is None:
            chunk_idx, file_idx = 0, 0
            latest_num_frames = 0
        else:
            latest_ep = self.meta.episodes[-1]
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)
            latest_num_frames = get_parquet_num_frames(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.meta.data_files_size_in_mb:
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                latest_num_frames = 0
            else:
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)
                del latest_df
                gc.collect()

        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.meta.image_keys) > 0:
            to_parquet_with_hf_images(df, path)
        else:
            df.to_parquet(path)

        self.hf_dataset = None
        return {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": latest_num_frames,
            "dataset_to_index": latest_num_frames + ep_num_frames,
        }

    dataset._save_episode_data = types.MethodType(_save_episode_data, dataset)


def _patch_save_episode_metadata(metadata: Any) -> None:
    def _save_episode_metadata(self, episode_dict: dict) -> None:
        import gc

        import datasets
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import DEFAULT_EPISODES_PATH
        from lerobot.datasets.lerobot_dataset import get_hf_dataset_size_in_mb
        from lerobot.datasets.lerobot_dataset import get_parquet_file_size_in_mb
        from lerobot.datasets.lerobot_dataset import load_episodes
        from lerobot.datasets.lerobot_dataset import update_chunk_file_indices

        episode_dict = {key: [value] for key, value in episode_dict.items()}
        ep_dataset = datasets.Dataset.from_dict(episode_dict)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        df = pd.DataFrame(ep_dataset)
        num_frames = episode_dict["length"][0]

        if self.episodes is None:
            chunk_idx, file_idx = 0, 0
            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [0]
            df["dataset_to_index"] = [num_frames]
        else:
            latest_ep = self.episodes[-1]
            chunk_idx = latest_ep["meta/episodes/chunk_index"]
            file_idx = latest_ep["meta/episodes/file_index"]

            latest_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.data_files_size_in_mb:
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)

            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [latest_ep["dataset_to_index"]]
            df["dataset_to_index"] = [latest_ep["dataset_to_index"] + num_frames]

            if latest_size_in_mb + ep_size_in_mb < self.data_files_size_in_mb:
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)
                del latest_df
                gc.collect()

        path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.episodes = load_episodes(self.root)

    metadata._save_episode_metadata = types.MethodType(_save_episode_metadata, metadata)


def convert_task_hdf5_to_lerobot(
    task_name: str,
    hdf5_paths: Sequence[str | Path],
    output_root: str | Path,
    *,
    repo_id_prefix: str = "",
    mode: Literal["video", "image"] = "video",
    fps: int = 20,
    robot_type: str = "aloha",
    overwrite: bool = False,
    max_episodes_per_task: int | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    if mode not in {"video", "image"}:
        raise ValueError(f"Unsupported mode={mode!r}. Expected 'video' or 'image'.")

    selected_paths = [Path(path).expanduser().resolve() for path in hdf5_paths]
    selected_paths = sorted(selected_paths, key=_natural_sort_key)
    if max_episodes_per_task is not None:
        selected_paths = selected_paths[: max(0, int(max_episodes_per_task))]
    if not selected_paths:
        raise ValueError(f"No HDF5 episodes found for task {task_name!r}.")

    schema = infer_hdf5_feature_schema(selected_paths)
    features = _build_lerobot_features(schema, mode, fps=int(fps))

    repo_name = _sanitize_name(f"{repo_id_prefix}{task_name}")
    output_root = Path(output_root).expanduser().resolve()
    dataset_root = output_root / repo_name
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {dataset_root}. Pass overwrite=True to rebuild it."
            )
        shutil.rmtree(dataset_root)

    LeRobotDataset = _load_lerobot_dataset_cls()
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=dataset_root,
        fps=int(fps),
        robot_type=str(robot_type),
        features=features,
        use_videos=(mode == "video"),
        video_backend="pyav",
    )
    if mode == "video":
        # Force one encoded video file per episode. This avoids LeRobot's video concatenation path,
        # which leaves transient NFS files behind on this machine and causes later episodes to fail.
        dataset.meta.info["video_files_size_in_mb"] = 0
        _patch_video_tempdir(dataset)
        _patch_save_episode_data(dataset)
        _patch_save_episode_metadata(dataset.meta)

    warnings: list[dict[str, Any]] = []
    written_episodes = 0
    written_frames = 0
    task_start_time = time.perf_counter()

    if num_workers <= 1:
        raise ValueError(
            f"num_workers must be > 1 for parallel processing. Got num_workers={num_workers}. "
            f"Please set num_workers_per_task to a value > 1."
        )

    print(
        f"[phase0] task={task_name} episodes={len(selected_paths)} output={dataset_root} "
        f"mode={mode} fps={fps} num_workers={num_workers}",
        flush=True,
    )

    global_frame_index = 0

    try:
        # Parallel processing mode
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from collections import OrderedDict
        
        # Prepare worker arguments.
        worker_args = [
            (hdf5_path, episode_pos, task_name, schema, fps)
            for episode_pos, hdf5_path in enumerate(selected_paths, start=1)
        ]
        
        episode_results: OrderedDict[int, EpisodeData] = OrderedDict()
        
        print(
            f"[phase0] task={task_name} starting parallel processing with {num_workers} workers",
            flush=True,
        )
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks.
            future_to_episode = {
                executor.submit(_load_episode_worker, *args): args[1]  # args[1] is episode_pos
                for args in worker_args
            }
            
            # Collect results as they finish; writes must be ordered by episode_pos.
            completed_count = 0
            for future in as_completed(future_to_episode):
                episode_pos = future_to_episode[future]
                completed_count += 1
                # Propagate exceptions without catching them.
                result = future.result()
                episode_results[episode_pos] = result
                print(
                    f"[phase0] task={task_name} episode={episode_pos}/{len(selected_paths)} "
                    f"loaded ({completed_count}/{len(selected_paths)})",
                    flush=True,
                )
        
        # Write in episode_pos order to preserve episode ordering.
        for episode_pos in sorted(episode_results.keys()):
            episode_data = episode_results[episode_pos]
            episode_start_time = time.perf_counter()
            print(
                f"[phase0] task={task_name} episode={episode_data.episode_pos}/{len(selected_paths)} "
                f"file={episode_data.hdf5_path.name} writing",
                flush=True,
            )
            
            # Record warnings about mismatched lengths.
            if episode_data.length_mismatch_warning is not None:
                warnings.append(episode_data.length_mismatch_warning)
            
            # Write directly without catching exceptions.
            for frame_index in range(episode_data.effective_length):
                dataset.add_frame({
                    "observation.state": episode_data.state[frame_index],
                    "observation.images.cam_high": episode_data.images["cam_high"][frame_index],
                    "observation.images.cam_left_wrist": episode_data.images["cam_left_wrist"][frame_index],
                    "observation.images.cam_right_wrist": episode_data.images["cam_right_wrist"][frame_index],
                    "action": episode_data.action[frame_index],
                    "reward": np.asarray([int(episode_data.rewards[frame_index])], dtype=np.uint8),
                    "result": np.asarray([episode_data.resolved_result], dtype=np.uint8),
                    "episode_success": np.asarray(
                        [int(bool(episode_data.episode_success[frame_index]))], dtype=np.uint8
                    ),
                    "episode_length": np.asarray([episode_data.effective_length], dtype=np.int64),
                    "timestep": np.asarray([int(episode_data.timesteps[frame_index])], dtype=np.int64),
                    "task_id": np.asarray([episode_data.task_id_value], dtype=np.int64),
                    "seed": np.asarray([episode_data.seed_value], dtype=np.int64),
                    "prompt": episode_data.prompts[frame_index],
                    # "timestamp": np.asarray([float(frame_index) / float(fps)], dtype=np.float32),
                    # "frame_index": np.asarray([frame_index], dtype=np.int64),
                    # "episode_index": np.asarray([episode_data.episode_pos - 1], dtype=np.int64),
                    # "index": np.asarray([global_frame_index], dtype=np.int64),
                    # "task_index": np.asarray([episode_data.task_id_value], dtype=np.int64),
                    "task": episode_data.task_text,
                })
                global_frame_index += 1
            
            dataset.hf_dataset = None
            dataset.save_episode()
            written_episodes += 1
            written_frames += episode_data.effective_length
            
            print(
                f"[phase0] task={task_name} episode={episode_data.episode_pos}/{len(selected_paths)} "
                f"frames={episode_data.effective_length} elapsed={time.perf_counter() - episode_start_time:.2f}s",
                flush=True,
            )
        
    finally:
        _finalize_dataset(dataset)

    manifest = {
        "task_name": task_name,
        "input_files": [str(path) for path in selected_paths],
        "input_root": str(selected_paths[0].parent) if selected_paths else "",
        "output_dataset_root": str(dataset_root),
        "repo_name": repo_name,
        "mode": mode,
        "fps": int(fps),
        "robot_type": str(robot_type),
        "episodes_seen": len(selected_paths),
        "episodes_written": written_episodes,
        "frames_written": written_frames,
        "warnings": warnings,
        "schema": asdict(schema),
    }

    manifest_path = dataset_root / "conversion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fout:
        json.dump(manifest, fout, indent=2, ensure_ascii=False)
        fout.write("\n")

    print(
        f"[phase0] task={task_name} done written_episodes={written_episodes} "
        f"written_frames={written_frames} elapsed={time.perf_counter() - task_start_time:.2f}s",
        flush=True,
    )

    return manifest


def convert_robotwin_run_to_lerobot(
    input_root: str | Path,
    output_root: str | Path,
    *,
    repo_id_prefix: str = "",
    mode: Literal["video", "image"] = "video",
    fps: int = 20,
    robot_type: str = "aloha",
    overwrite: bool = False,
    max_episodes_per_task: int | None = None,
) -> list[dict[str, Any]]:
    task_to_files = scan_task_hdf5_dirs(input_root)
    if not task_to_files:
        raise ValueError(f"No task subdirectories with HDF5 episodes were found under {input_root}.")

    manifests = []
    for task_name, hdf5_paths in task_to_files.items():
        manifest = convert_task_hdf5_to_lerobot(
            task_name,
            hdf5_paths,
            output_root,
            repo_id_prefix=repo_id_prefix,
            mode=mode,
            fps=fps,
            robot_type=robot_type,
            overwrite=overwrite,
            max_episodes_per_task=max_episodes_per_task,
        )
        manifests.append(manifest)
    return manifests


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Robotwin rollout HDF5 episodes into LeRobot v3 datasets for qwenvl."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Robotwin rollout run root directory.")
    parser.add_argument("--output_root", type=str, required=True, help="Directory where converted datasets are created.")
    parser.add_argument(
        "--repo_id_prefix",
        type=str,
        default="",
        help="Optional prefix prepended to each task dataset directory name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("video", "image"),
        default="video",
        help="LeRobot image storage mode.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Dataset FPS written into LeRobot metadata.")
    parser.add_argument("--robot_type", type=str, default="aloha", help="Robot type stored in LeRobot metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted task datasets.")
    parser.add_argument(
        "--max_episodes_per_task",
        type=int,
        default=None,
        help="Optional limit on the number of HDF5 episodes converted per task.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifests = convert_robotwin_run_to_lerobot(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_id_prefix=args.repo_id_prefix,
        mode=args.mode,
        fps=args.fps,
        robot_type=args.robot_type,
        overwrite=args.overwrite,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    print(json.dumps(manifests, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "CAMERA_NAMES",
    "HDF5FeatureSchema",
    "convert_robotwin_run_to_lerobot",
    "convert_task_hdf5_to_lerobot",
    "infer_hdf5_feature_schema",
    "main",
    "scan_task_hdf5_dirs",
]


if __name__ == "__main__":
    raise SystemExit(main())
