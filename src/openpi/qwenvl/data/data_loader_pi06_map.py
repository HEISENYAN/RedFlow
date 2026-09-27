from __future__ import annotations

import bisect
import os
from collections import OrderedDict
from typing import Any
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from openpi.qwenvl.data._lerobot_dataset import load_episode_records
from openpi.qwenvl.data._lerobot_dataset import open_lerobot_dataset
from openpi.qwenvl.data._lerobot_dataset import split_train_val
from openpi.qwenvl.data._lerobot_dataset import tensor_to_pil_image
from openpi.qwenvl.modeling._prompt_templates import VALUE_PROMPT_TEMPLATE

FRAME_CACHE_SIZE = 64
EPISODE_CACHE_SLOTS = 4


def _to_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like value, got sequence={value}")
        return _to_scalar(value[0])
    return float(value)


def _to_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "success", "yes", "y"}:
            return True
        if lowered in {"0", "false", "fail", "failure", "no", "n"}:
            return False
    return bool(value)


class MapPi06Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 42,
        val_ratio: float = 0.1,
        camera_names: Optional[list[str]] = None,
        value_tokenizer=None,
        split: str = "train",
        max_episodes: Optional[int] = None,
        language_instruction: str = "perform the task",
        value_source: str = "dataset",
        value_column: str = "value",
    ) -> None:
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.seed = int(seed)
        self.camera_names = camera_names or ["cam_high"]
        self.value_tokenizer = value_tokenizer
        self.split = str(split)
        self.language_instruction = language_instruction
        self.value_source = str(value_source).strip().lower()
        self.value_column = str(value_column)
        if self.value_source not in {"dataset", "episode_progress"}:
            raise ValueError(f"Unsupported value_source={value_source!r}. Expected 'dataset' or 'episode_progress'.")

        self._lerobot_dataset = None
        self._init_pid = None

        self.episodes_meta = load_episode_records(
            self.dataset_dir,
            max_episodes=max_episodes,
        )
        self.episodes_meta = split_train_val(
            self.episodes_meta,
            val_ratio=val_ratio,
            seed=self.seed,
            split=split,
        )
        self._global_step_prefix: list[int] = []
        self._build_indices()

        self._episode_caches: dict[int, OrderedDict[int, dict[str, Image.Image]]] = {}
        self._episode_value_caches: dict[int, OrderedDict[int, float]] = {}
        self._episode_success_cache: dict[int, Optional[bool]] = {}
        self._episode_cache_order: OrderedDict[int, None] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lerobot_dataset"] = None
        state["_init_pid"] = None
        state["_episode_caches"] = {}
        state["_episode_value_caches"] = {}
        state["_episode_success_cache"] = {}
        state["_episode_cache_order"] = OrderedDict()
        return state

    def _build_indices(self):
        total = 0
        for episode_meta in self.episodes_meta:
            total += int(episode_meta["length"])
            self._global_step_prefix.append(total)

    def __len__(self):
        return self._global_step_prefix[-1] if self._global_step_prefix else 0

    def _get_lerobot_dataset(self):
        pid = os.getpid()
        if self._lerobot_dataset is None or self._init_pid != pid:
            self._lerobot_dataset = open_lerobot_dataset(self.dataset_dir)
            self._init_pid = pid
        return self._lerobot_dataset

    def _touch_episode_cache(self, episode_idx: int):
        if episode_idx in self._episode_cache_order:
            self._episode_cache_order.move_to_end(episode_idx)
            return
        self._episode_cache_order[episode_idx] = None
        if len(self._episode_cache_order) > EPISODE_CACHE_SLOTS:
            old_episode_idx, _ = self._episode_cache_order.popitem(last=False)
            self._episode_caches.pop(old_episode_idx, None)
            self._episode_value_caches.pop(old_episode_idx, None)
            self._episode_success_cache.pop(old_episode_idx, None)

    def _get_raw_row(self, episode_meta: dict[str, Any], step_idx: int):
        return self._get_lerobot_dataset()[int(episode_meta["global_start_index"]) + int(step_idx)]

    def _get_step_images(self, episode_meta: dict[str, Any], step_idx: int) -> dict[str, Image.Image]:
        episode_idx = int(episode_meta["episode_idx"])
        self._touch_episode_cache(episode_idx)
        frame_cache = self._episode_caches.setdefault(episode_idx, OrderedDict())
        if step_idx in frame_cache:
            frame_cache.move_to_end(step_idx)
            return frame_cache[step_idx]

        raw_row = self._get_raw_row(episode_meta, step_idx)
        images = {}
        for camera_name in self.camera_names:
            image_key = f"observation.images.{camera_name}"
            if image_key not in raw_row:
                raise KeyError(
                    f"Missing required camera data {image_key!r} in dataset={self.dataset_dir}, "
                    f"episode={episode_idx}, step={step_idx}"
                )
            images[camera_name] = tensor_to_pil_image(raw_row[image_key])

        frame_cache[step_idx] = images
        if len(frame_cache) > FRAME_CACHE_SIZE:
            frame_cache.popitem(last=False)
        return images

    def _resolve_episode_success(self, episode_meta: dict[str, Any], step_idx: int) -> bool:
        episode_idx = int(episode_meta["episode_idx"])
        cached = self._episode_success_cache.get(episode_idx)
        if cached is not None:
            return bool(cached)

        success_meta = episode_meta.get("success")
        if success_meta is not None:
            success = _to_bool(success_meta)
            self._episode_success_cache[episode_idx] = success
            return success

        raw_row = self._get_raw_row(episode_meta, step_idx)
        for key in ("episode_success", "result", "success"):
            if key in raw_row:
                success = _to_bool(raw_row[key])
                self._episode_success_cache[episode_idx] = success
                return success

        self._episode_success_cache[episode_idx] = True
        return True

    def _get_canonical_value(self, episode_meta: dict[str, Any], step_idx: int) -> float:
        if not self.value_column:
            raise ValueError("value_column must be configured for strict pi06 training")
        episode_idx = int(episode_meta["episode_idx"])
        self._touch_episode_cache(episode_idx)
        value_cache = self._episode_value_caches.setdefault(episode_idx, OrderedDict())
        if step_idx in value_cache:
            value_cache.move_to_end(step_idx)
            return float(value_cache[step_idx])

        raw_row = self._get_raw_row(episode_meta, step_idx)
        if self.value_source == "dataset":
            if self.value_column not in raw_row:
                raise KeyError(
                    f"Missing required value column {self.value_column!r} in dataset={self.dataset_dir}, "
                    f"episode={episode_idx}, step={step_idx}"
                )
            canonical_value = np.clip(_to_scalar(raw_row[self.value_column]), 0.0, 1.0)
        else:
            episode_length = max(1, int(episode_meta["length"]))
            canonical_value = np.clip(float(step_idx) / float(episode_length), 0.0, 1.0)
        value_cache[step_idx] = float(canonical_value)
        if len(value_cache) > FRAME_CACHE_SIZE:
            value_cache.popitem(last=False)
        return float(canonical_value)

    def _format_value_text(self, model_value: float) -> str:
        if self.value_tokenizer is not None:
            return str(self.value_tokenizer(np.array([model_value], dtype=np.float32)))
        return str(model_value)

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = bisect.bisect_right(self._global_step_prefix, index)
        prefix_before_episode = 0 if episode_pos == 0 else self._global_step_prefix[episode_pos - 1]
        step_idx = index - prefix_before_episode
        return episode_pos, step_idx

    def __getitem__(self, index: int):
        episode_pos, step_idx = self._resolve_index(int(index))
        episode_meta = self.episodes_meta[episode_pos]
        step_images = self._get_step_images(episode_meta, step_idx)
        image_list = [step_images[camera_name] for camera_name in self.camera_names]

        canonical_value = self._get_canonical_value(episode_meta, step_idx)
        model_value = float(canonical_value)

        return {
            "conversations": [
                {"from": "human", "value": VALUE_PROMPT_TEMPLATE.format(instruction=episode_meta["instruction"])},
                {"from": "gpt", "value": self._format_value_text(model_value)},
            ],
            "image": image_list,
            "meta_value": canonical_value,
            "meta_model_value": model_value,
            "meta_ep_idx": int(episode_meta["episode_idx"]),
            "meta_t": int(step_idx),
            "meta_episode_len": int(episode_meta["length"]),
            "meta_dataset_dir": self.dataset_dir,
        }
