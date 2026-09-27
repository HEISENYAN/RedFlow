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
from openpi.qwenvl.modeling._prompt_templates import PAIR_PROMPT_TEMPLATE

FRAME_CACHE_SIZE = 64
EPISODE_CACHE_SLOTS = 4


def _to_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like target, got sequence={value}")
        return _to_scalar(value[0])
    return float(value)


class MapPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        seed: int = 42,
        val_ratio: float = 0.1,
        camera_names: Optional[list[str]] = None,
        split: str = "train",
        max_episodes: Optional[int] = None,
        pair_add_backward: bool = True,
        pair_add_zero_anchor: Optional[bool] = True,
        pair_target_column: Optional[str] = None,
        pair_prompt_style: str = "current_history_delta",
    ) -> None:
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.seed = int(seed)
        self.camera_names = camera_names or ["cam_high"]
        self.split = str(split)
        self.max_episodes = max_episodes
        self.pair_add_backward = bool(pair_add_backward)
        self.pair_add_zero_anchor = True if pair_add_zero_anchor is None else bool(pair_add_zero_anchor)
        self.pair_target_column = str(pair_target_column).strip() if pair_target_column else None
        self.pair_prompt_style = str(pair_prompt_style)
        if self.pair_prompt_style != "current_history_delta":
            raise ValueError(f"Unsupported pair_prompt_style={self.pair_prompt_style!r}")

        self._lerobot_dataset = None
        self._init_pid = None

        self.episodes_meta = load_episode_records(self.dataset_dir, max_episodes=max_episodes)
        self.episodes_meta = split_train_val(
            self.episodes_meta,
            val_ratio=val_ratio,
            seed=self.seed,
            split=split,
        )
        self._episode_pair_prefix: list[list[int]] = []
        self._global_pair_prefix: list[int] = []
        self._build_indices()

        self._episode_caches: dict[int, OrderedDict[int, dict[str, Image.Image]]] = {}
        self._episode_target_caches: dict[int, OrderedDict[int, float]] = {}
        self._episode_cache_order: OrderedDict[int, None] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_lerobot_dataset"] = None
        state["_init_pid"] = None
        state["_episode_caches"] = {}
        state["_episode_target_caches"] = {}
        state["_episode_cache_order"] = OrderedDict()
        return state

    def _pair_count_for_t(self, step_idx: int) -> int:
        if step_idx <= 0:
            return 0
        count = 2 if self.pair_add_backward else 1
        if self.pair_add_zero_anchor:
            count += 2 if self.pair_add_backward else 1
        return count

    def _build_indices(self):
        total = 0
        for episode_meta in self.episodes_meta:
            prefix = [0]
            running = 0
            for step_idx in range(int(episode_meta["length"])):
                running += self._pair_count_for_t(step_idx)
                prefix.append(running)
            self._episode_pair_prefix.append(prefix)
            total += running
            self._global_pair_prefix.append(total)

    def __len__(self):
        return self._global_pair_prefix[-1] if self._global_pair_prefix else 0

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
            self._episode_target_caches.pop(old_episode_idx, None)

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

    def _get_step_target_value(self, episode_meta: dict[str, Any], step_idx: int) -> float:
        if not self.pair_target_column:
            raise RuntimeError("pair_target_column is not configured")
        episode_idx = int(episode_meta["episode_idx"])
        self._touch_episode_cache(episode_idx)
        value_cache = self._episode_target_caches.setdefault(episode_idx, OrderedDict())
        if step_idx in value_cache:
            value_cache.move_to_end(step_idx)
            return float(value_cache[step_idx])

        raw_row = self._get_raw_row(episode_meta, step_idx)
        if self.pair_target_column not in raw_row:
            raise KeyError(
                f"Missing pair target column {self.pair_target_column!r} in dataset={self.dataset_dir}, "
                f"episode={episode_idx}, step={step_idx}"
            )
        target_value = _to_scalar(raw_row[self.pair_target_column])
        value_cache[step_idx] = target_value
        if len(value_cache) > FRAME_CACHE_SIZE:
            value_cache.popitem(last=False)
        return float(target_value)

    def _sample_history_step(self, episode_idx: int, current_step: int) -> int:
        if current_step <= 0:
            raise ValueError(f"current_step must be > 0, got {current_step}")
        rng = np.random.default_rng(self.seed + int(episode_idx) * 9973 + int(current_step) * 101)
        return int(rng.integers(0, current_step))

    def _variants_for_t(self, episode_meta: dict[str, Any], step_idx: int) -> list[tuple[int, int, float, str]]:
        if step_idx <= 0:
            return []

        history_step = self._sample_history_step(int(episode_meta["episode_idx"]), step_idx)
        if self.pair_target_column:
            delta = np.clip(
                self._get_step_target_value(episode_meta, step_idx)
                - self._get_step_target_value(episode_meta, history_step),
                -1.0,
                1.0,
            )
        else:
            episode_length = max(1, int(episode_meta["length"]))
            delta = np.clip((step_idx - history_step) / episode_length, -1.0, 1.0)

        variants = [(step_idx, history_step, float(delta), "current_history_fwd")]
        if self.pair_add_backward:
            variants.append((history_step, step_idx, float(-delta), "current_history_bwd"))

        if self.pair_add_zero_anchor:
            if self.pair_target_column:
                zero_delta = np.clip(
                    self._get_step_target_value(episode_meta, step_idx)
                    - self._get_step_target_value(episode_meta, 0),
                    -1.0,
                    1.0,
                )
            else:
                episode_length = max(1, int(episode_meta["length"]))
                zero_delta = np.clip(step_idx / episode_length, -1.0, 1.0)
            variants.append((step_idx, 0, float(zero_delta), "current_zero_fwd"))
            if self.pair_add_backward:
                variants.append((0, step_idx, float(-zero_delta), "current_zero_bwd"))
        return variants

    def _resolve_index(self, index: int) -> tuple[int, int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_pos = bisect.bisect_right(self._global_pair_prefix, index)
        prefix_before_episode = 0 if episode_pos == 0 else self._global_pair_prefix[episode_pos - 1]
        local_pair_idx = index - prefix_before_episode
        timestep_prefix = self._episode_pair_prefix[episode_pos]
        step_idx = bisect.bisect_right(timestep_prefix, local_pair_idx) - 1
        pair_idx = local_pair_idx - timestep_prefix[step_idx]
        return episode_pos, step_idx, pair_idx

    def __getitem__(self, index: int):
        episode_pos, step_idx, pair_idx = self._resolve_index(int(index))
        episode_meta = self.episodes_meta[episode_pos]
        variants = self._variants_for_t(episode_meta, step_idx)
        if not variants:
            raise RuntimeError(
                f"Resolved pair index points to empty timestep: dataset={self.dataset_dir}, "
                f"episode={episode_meta['episode_idx']}, step={step_idx}"
            )

        current_step, history_step, delta, pair_type = variants[pair_idx]
        current_images = self._get_step_images(episode_meta, current_step)
        history_images = self._get_step_images(episode_meta, history_step)
        image_list = [current_images[camera_name] for camera_name in self.camera_names]
        image_list.extend(history_images[camera_name] for camera_name in self.camera_names)

        return {
            "conversations": [
                {"from": "human", "value": PAIR_PROMPT_TEMPLATE.format(instruction=episode_meta["instruction"])},
                {"from": "gpt", "value": ""},
            ],
            "image": image_list,
            "delta_label": float(delta),
            "pair_type": pair_type,
            "t_group_weight": float(1.0 / len(variants)),
            "meta_ep_idx": int(episode_meta["episode_idx"]),
            "meta_t": int(step_idx),
            "meta_current": int(current_step),
            "meta_history": int(history_step),
            "meta_episode_len": int(episode_meta["length"]),
            "meta_dataset_dir": self.dataset_dir,
        }
