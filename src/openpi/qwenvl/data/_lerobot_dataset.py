from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Sequence

import datasets.features.features as hf_features
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

if "List" not in hf_features._FEATURE_TYPES:
    hf_features._FEATURE_TYPES["List"] = hf_features.Sequence

_CANONICAL_CAMERA_KEYS = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
TARGET_IMAGE_HEIGHT = 240
TARGET_IMAGE_WIDTH = 320
TARGET_IMAGE_SIZE = (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)


def extract_episode_instruction(episode_meta: dict[str, Any], episode_idx: int) -> str:
    raw_instruction = episode_meta.get("tasks")
    if isinstance(raw_instruction, list) and raw_instruction:
        raw_instruction = raw_instruction[0]
    if raw_instruction is None:
        raw_instruction = episode_meta.get("task")
    if raw_instruction is None or str(raw_instruction).strip() == "":
        raise ValueError(
            f"Episode {episode_idx} is missing task instruction metadata. "
            "Expected a non-empty `tasks[0]` or `task` value."
        )
    return str(raw_instruction)


def load_episode_records(
    dataset_dir: str,
    *,
    max_episodes: int | None = None,
) -> list[dict[str, Any]]:
    metadata = LeRobotDatasetMetadata(dataset_dir)
    raw_episodes = getattr(metadata, "episodes", None)
    if not raw_episodes:
        raise ValueError(f"LeRobotDatasetMetadata.episodes is empty for dataset_dir={dataset_dir}")

    if isinstance(raw_episodes, dict):
        ordered = sorted(raw_episodes.items(), key=lambda item: int(item[0]))
    else:
        ordered = list(enumerate(raw_episodes))
    if max_episodes is not None:
        ordered = ordered[: int(max_episodes)]

    episodes: list[dict[str, Any]] = []
    for ep_key, ep_meta in ordered:
        episode_idx = int(ep_key)
        episode_meta = dict(ep_meta)
        length = int(episode_meta["length"])
        dataset_from_index = episode_meta.get("dataset_from_index")
        dataset_to_index = episode_meta.get("dataset_to_index")
        if dataset_from_index is None or dataset_to_index is None:
            raise ValueError(
                f"Episode {episode_idx} is missing dataset row bounds in metadata: "
                "`dataset_from_index` and `dataset_to_index` are required."
            )
        dataset_from_index = int(dataset_from_index)
        dataset_to_index = int(dataset_to_index)
        dataset_length = dataset_to_index - dataset_from_index
        if dataset_length != length:
            raise ValueError(
                f"Episode {episode_idx} metadata mismatch: length={length} "
                f"but dataset_to_index - dataset_from_index = {dataset_length}."
            )
        episodes.append(
            {
                "episode_idx": episode_idx,
                "global_start_index": dataset_from_index,
                "length": length,
                "instruction": extract_episode_instruction(episode_meta, episode_idx),
                "success": episode_meta.get("success", episode_meta.get("result")),
            }
        )

    if not episodes:
        raise ValueError(f"No episode metadata found for dataset_dir={dataset_dir}")
    return episodes


def split_train_val(
    episodes: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    count = len(episodes)
    if count == 0:
        return []
    indices = np.arange(count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = int(count * val_ratio)
    if count >= 2:
        val_count = max(1, min(count - 1, val_count))
    else:
        val_count = min(count, val_count)
    selected = indices[:val_count] if split == "val" else indices[val_count:]
    return [episodes[i] for i in selected]


def open_lerobot_dataset(dataset_dir: str) -> LeRobotDataset:
    return LeRobotDataset(str(Path(dataset_dir).expanduser().resolve()), video_backend="pyav")


def load_lerobot_dataset(repo_root: str | Path) -> LeRobotDataset:
    return open_lerobot_dataset(str(Path(repo_root).expanduser().resolve()))


def load_lerobot_metadata(repo_root: str | Path) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(str(Path(repo_root).expanduser().resolve()))


def resize_pil_rgb_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.size != TARGET_IMAGE_SIZE:
        image = image.resize(TARGET_IMAGE_SIZE, resample=Image.BILINEAR)
    return image


def lerobot_image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return resize_pil_rgb_image(image)
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unexpected LeRobot image shape: {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return resize_pil_rgb_image(Image.fromarray(array, mode="RGB"))


def tensor_to_pil_image(image_tensor: Any) -> Image.Image:
    return lerobot_image_to_pil(image_tensor)


def pil_image_to_rgb_np(image: Image.Image) -> np.ndarray:
    return np.asarray(resize_pil_rgb_image(image), dtype=np.uint8)


def _safe_scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "item"):
        value = value.item()
    return value


def resolve_prompt(metadata: LeRobotDatasetMetadata, episode_idx: int, prompt_override: Optional[str]) -> str:
    if prompt_override:
        return str(prompt_override)
    return extract_episode_instruction(dict(metadata.episodes[int(episode_idx)]), int(episode_idx))


def _format_data_path(pattern: str, episode_idx: int, chunks_size: int) -> Path:
    chunk_index = int(episode_idx) // int(chunks_size)
    return Path(
        pattern.format(
            episode_chunk=chunk_index,
            episode_index=int(episode_idx),
            chunk_index=chunk_index,
            file_index=int(episode_idx),
        )
    )


def _episode_metadata_dict(metadata: LeRobotDatasetMetadata, episode_idx: int) -> dict[str, Any]:
    raw_episode = metadata.episodes[int(episode_idx)]
    return dict(raw_episode)


def resolve_parquet_path(repo_root: str | Path, metadata: LeRobotDatasetMetadata, episode_idx: int) -> Path:
    repo_path = Path(repo_root).expanduser().resolve()
    episode_meta = _episode_metadata_dict(metadata, episode_idx)
    if "data/chunk_index" in episode_meta and "data/file_index" in episode_meta:
        rel_path = Path(
            str(metadata.data_path).format(
                episode_chunk=int(episode_meta["data/chunk_index"]),
                episode_index=int(episode_idx),
                chunk_index=int(episode_meta["data/chunk_index"]),
                file_index=int(episode_meta["data/file_index"]),
            )
        )
    else:
        rel_path = _format_data_path(str(metadata.data_path), episode_idx, int(metadata.chunks_size))
    parquet_path = repo_path / rel_path
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet for episode {episode_idx}: {parquet_path}")
    return parquet_path


def resolve_episode_dataset_range(metadata: LeRobotDatasetMetadata, episode_idx: int) -> tuple[int, int]:
    episode_meta = _episode_metadata_dict(metadata, episode_idx)
    start = episode_meta.get("dataset_from_index")
    end = episode_meta.get("dataset_to_index")
    if start is None or end is None:
        raise ValueError(f"Episode {episode_idx} is missing dataset row bounds in metadata.")
    return int(start), int(end)


def resolve_episode_local_row_range(metadata: LeRobotDatasetMetadata, episode_idx: int) -> tuple[int, int]:
    episode_meta = _episode_metadata_dict(metadata, episode_idx)
    start, end = resolve_episode_dataset_range(metadata, episode_idx)
    chunk_index = episode_meta.get("data/chunk_index")
    file_index = episode_meta.get("data/file_index")
    if chunk_index is None or file_index is None:
        return start, end

    file_starts = []
    raw_episodes = metadata.episodes.values() if isinstance(metadata.episodes, dict) else metadata.episodes
    for raw_episode in raw_episodes:
        raw_dict = dict(raw_episode)
        if (
            raw_dict.get("data/chunk_index") == chunk_index
            and raw_dict.get("data/file_index") == file_index
            and raw_dict.get("dataset_from_index") is not None
        ):
            file_starts.append(int(raw_dict["dataset_from_index"]))
    if not file_starts:
        raise ValueError(f"Could not resolve local parquet row range for episode {episode_idx}")
    base_start = min(file_starts)
    return start - base_start, end - base_start


def read_frame_rows(src_parquet: Path, *, row_start: int | None = None, row_end: int | None = None) -> list[dict[str, Any]]:
    schema = pq.read_schema(src_parquet)
    columns = ["frame_index"]
    for key in ("result", "episode_success", "success"):
        if key in schema.names:
            columns.append(key)
    if "frame_index" in schema.names:
        table = pq.read_table(src_parquet, columns=columns)
        if row_start is not None or row_end is not None:
            start = 0 if row_start is None else int(row_start)
            end = table.num_rows if row_end is None else int(row_end)
            table = table.slice(start, max(0, end - start))
        frame_indices = [_safe_scalar(value) for value in table["frame_index"].to_pylist()]
    else:
        table = None
        total_rows = int(pq.read_metadata(src_parquet).num_rows)
        start = 0 if row_start is None else int(row_start)
        end = total_rows if row_end is None else int(row_end)
        frame_indices = list(range(start, end))
    frame_rows: list[dict[str, Any]] = []
    for index, frame_index in enumerate(frame_indices):
        frame_row = {"dataset_frame_index": int(frame_index)}
        if table is not None:
            for key in columns:
                if key == "frame_index":
                    continue
                frame_row[key] = _safe_scalar(table[key][index])
        frame_rows.append(frame_row)
    return frame_rows


def attach_frame_metadata(rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != len(frame_rows):
        raise ValueError(f"Prediction row count mismatch: predictions={len(rows)} frame_rows={len(frame_rows)}")
    out = []
    for row, frame_meta in zip(rows, frame_rows, strict=True):
        enriched = dict(row)
        enriched.update(frame_meta)
        enriched["dataset_future_frame_index"] = int(frame_rows[int(row["future_frame_idx"])]["dataset_frame_index"])
        out.append(enriched)
    return out


def _image_to_rgb_np(image: Any) -> np.ndarray:
    return pil_image_to_rgb_np(lerobot_image_to_pil(image))


def sampled_episode_row_indices(
    metadata: LeRobotDatasetMetadata,
    episode_idx: int,
    *,
    frame_interval: int = 1,
) -> list[int]:
    start, end = resolve_episode_dataset_range(metadata, episode_idx)
    stride = max(1, int(frame_interval))
    return [row_index for local_idx, row_index in enumerate(range(start, end)) if local_idx % stride == 0]


def resolve_camera_feature_keys(camera_names: Sequence[str]) -> tuple[str, ...]:
    if len(camera_names) < 1:
        raise ValueError("Expected at least one camera name.")
    keys = []
    for camera_name in camera_names:
        if camera_name not in _CANONICAL_CAMERA_KEYS:
            raise ValueError(
                f"Unsupported camera name {camera_name!r}. Expected one of {sorted(_CANONICAL_CAMERA_KEYS)}"
            )
        keys.append(_CANONICAL_CAMERA_KEYS[camera_name])
    return tuple(keys)


def load_episode_frame_triplets(
    dataset: LeRobotDataset,
    metadata: LeRobotDatasetMetadata,
    episode_idx: int,
    camera_names: Sequence[str],
    *,
    frame_interval: int = 1,
) -> list[tuple[Image.Image, ...]]:
    camera_keys = resolve_camera_feature_keys(camera_names)
    sampled_triplets: list[tuple[Image.Image, ...]] = []
    for row_index in sampled_episode_row_indices(metadata, episode_idx, frame_interval=frame_interval):
        row = dataset[row_index]
        sampled_triplets.append(
            tuple(lerobot_image_to_pil(row[camera_key]) for camera_key in camera_keys)
        )
    return sampled_triplets


def iter_episode_frame_triplet_chunks(
    dataset: LeRobotDataset,
    metadata: LeRobotDatasetMetadata,
    episode_idx: int,
    camera_names: Sequence[str],
    *,
    frame_interval: int = 1,
    chunk_size: int = 128,
) -> Iterator[list[tuple[Image.Image, ...]]]:
    camera_keys = resolve_camera_feature_keys(camera_names)
    chunk_capacity = max(1, int(chunk_size))
    chunk: list[tuple[Image.Image, ...]] = []
    for row_index in sampled_episode_row_indices(metadata, episode_idx, frame_interval=frame_interval):
        row = dataset[row_index]
        chunk.append(tuple(lerobot_image_to_pil(row[camera_key]) for camera_key in camera_keys))
        if len(chunk) >= chunk_capacity:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def iter_pair_episode_triplet_chunks(
    dataset: LeRobotDataset,
    metadata: LeRobotDatasetMetadata,
    episode_idx: int,
    camera_names: Sequence[str],
    *,
    frame_interval: int = 1,
    relative_interval: int = 50,
    chunk_size: int = 128,
) -> Iterator[tuple[list[tuple[Image.Image, ...]], list[tuple[Image.Image, ...]], list[tuple[Image.Image, ...]]]]:
    sampled_row_indices = sampled_episode_row_indices(metadata, episode_idx, frame_interval=frame_interval)
    if not sampled_row_indices:
        return
    camera_keys = resolve_camera_feature_keys(camera_names)
    chunk_capacity = max(1, int(chunk_size))
    future_offset = max(1, int(relative_interval))

    def load_triplet(row_index: int, cache: dict[int, tuple[Image.Image, ...]]):
        cached = cache.get(row_index)
        if cached is not None:
            return cached
        row = dataset[row_index]
        triplet = tuple(lerobot_image_to_pil(row[camera_key]) for camera_key in camera_keys)
        cache[row_index] = triplet
        return triplet

    reference_row_index = sampled_row_indices[0]
    for chunk_start in range(0, len(sampled_row_indices), chunk_capacity):
        chunk_end = min(chunk_start + chunk_capacity, len(sampled_row_indices))
        chunk_cache: dict[int, tuple[Image.Image, ...]] = {}
        reference_triplet = load_triplet(reference_row_index, chunk_cache)
        current_chunk: list[tuple[Image.Image, ...]] = []
        reference_chunk: list[tuple[Image.Image, ...]] = []
        future_chunk: list[tuple[Image.Image, ...]] = []
        for local_idx in range(chunk_start, chunk_end):
            current_triplet = load_triplet(sampled_row_indices[local_idx], chunk_cache)
            future_idx = min(local_idx + future_offset, len(sampled_row_indices) - 1)
            future_triplet = load_triplet(sampled_row_indices[future_idx], chunk_cache)
            current_chunk.append(current_triplet)
            reference_chunk.append(reference_triplet)
            future_chunk.append(future_triplet)
        yield current_chunk, reference_chunk, future_chunk
