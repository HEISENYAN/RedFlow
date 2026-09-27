from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from openpi.qwenvl.data._lerobot_dataset import load_lerobot_metadata
from openpi.qwenvl.data._lerobot_dataset import resize_pil_rgb_image
from openpi.qwenvl.data._lerobot_dataset import resolve_prompt
from openpi.qwenvl.modeling.pi06_inference import assert_strict_checkpoint_dir
from openpi.qwenvl.modeling.pi06_inference import load_pi06_inference_components
from openpi.qwenvl.modeling.pi06_inference import predict_pi06_values

CAMERA_NAMES = ("cam_high",)


def _to_uint8_image(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    if array.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if float(array.min()) < 0.0:
        array = (array + 1.0) / 2.0
    elif float(array.max()) > 1.5:
        array = array / 255.0
    array = np.clip(array * 255.0, 0.0, 255.0)
    return array.astype(np.uint8)


def _view_to_pil(view: np.ndarray | torch.Tensor) -> Image.Image:
    if torch.is_tensor(view):
        view = view.detach().cpu().numpy()
    array = np.asarray(view)
    if array.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape={array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape={array.shape}")
    return resize_pil_rgb_image(Image.fromarray(_to_uint8_image(array), mode="RGB"))


def _normalize_episode_indices(episode_index: Any) -> list[int]:
    if torch.is_tensor(episode_index):
        episode_index = episode_index.detach().cpu().numpy()
    array = np.asarray(episode_index)
    if array.ndim == 0:
        return [int(array.item())]
    if array.ndim >= 2:
        array = array.reshape(array.shape[0], -1)[:, 0]
    return [int(item) for item in array.tolist()]


def _obs_batch_to_views(obs_batch: torch.Tensor | np.ndarray) -> list[tuple[Image.Image, ...]]:
    if torch.is_tensor(obs_batch):
        array = obs_batch.detach().cpu().numpy()
    else:
        array = np.asarray(obs_batch)
    if array.ndim == 4:
        array = np.expand_dims(array, axis=0)
    if array.ndim != 5:
        raise ValueError(f"Expected observation batch with 5 dims, got shape={array.shape}")

    views = []
    for example in array:
        if example.shape[0] < 1:
            raise ValueError(f"QwenVL value function expects at least one view, got per-sample shape={example.shape}")
        views.append(tuple(_view_to_pil(view) for view in example))
    return views


class ValueFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.value_function_path:
            raise ValueError("config.value_function_path must be set to use QwenVL value function inference.")

        repo_id = getattr(config.data, "repo_id", None)
        if repo_id is None or isinstance(repo_id, (list, tuple)):
            raise ValueError(
                "QwenVL value function runtime requires config.data.repo_id to be a single LeRobot dataset path."
            )

        self.model_type = str(getattr(config, "value_function_model_type", "pi06")).lower()
        if self.model_type != "pi06":
            raise NotImplementedError(f"ValueFunction currently only supports pi06, got model_type={self.model_type!r}")
        self.batch_size = max(1, min(int(getattr(config, "batch_size", 8)), 16))
        self.repo_id = str(repo_id)
        self.metadata = load_lerobot_metadata(self.repo_id)
        self.components = load_pi06_inference_components(
            assert_strict_checkpoint_dir(Path(config.value_function_path).resolve()),
            bf16=torch.cuda.is_available(),
        )
        self.model = self.components["model"]
        self.processor = self.components["processor"]
        self.value_tokenizer = self.components["value_tokenizer"]

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._prompt_cache: dict[int, str] = {}

    def _resolve_prompt_for_episode(self, episode_idx: int) -> str:
        if episode_idx in self._prompt_cache:
            return self._prompt_cache[episode_idx]
        prompt = resolve_prompt(self.metadata, episode_idx, None)
        self._prompt_cache[episode_idx] = prompt
        return prompt

    def forward(self, obs_batch, episode_index):
        return self.predict_value(obs_batch, episode_index)

    @torch.inference_mode()
    def predict_value(self, obs_batch, episode_index):
        device = next(self.model.parameters()).device
        self.components["device"] = device
        episode_indices = _normalize_episode_indices(episode_index)
        prompts = [self._resolve_prompt_for_episode(ep_idx) for ep_idx in episode_indices]
        current_views = _obs_batch_to_views(obs_batch)
        if len(current_views) != len(prompts):
            raise ValueError(
                f"Observation batch size and episode index size mismatch: "
                f"{len(current_views)} vs {len(prompts)}"
            )

        values = predict_pi06_values(self.components, prompts, current_views, batch_size=self.batch_size)
        return torch.as_tensor(values, device=device, dtype=torch.float32)
