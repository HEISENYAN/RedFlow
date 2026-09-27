from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    pair_mode: bool = field(default=False)
    pair_data_backend: str = field(default="map")
    pair_add_backward: bool = field(default=True)
    pair_add_zero_anchor: Optional[bool] = field(default=True)
    pair_target_column: Optional[str] = field(default=None)
    pair_prompt_style: str = field(default="current_history_delta")
    camera_names: str = field(default="cam_high")
    val_ratio: float = field(default=0.1)
    max_episodes: Optional[int] = field(default=None)
    base_interval: int = field(default=2)
    image_height: int = field(default=240)
    image_width: int = field(default=320)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = field(default=2.0)
    use_value_tokenizer: bool = field(default=False)
    value_tokenizer_bins: int = field(default=201)
    value_tokenizer_min: float = field(default=0.0)
    value_tokenizer_max: float = field(default=1.0)
    value_source: str = field(default="dataset")
    value_column: str = field(default="value")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length after chat templating."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    value_head_lr: Optional[float] = None
    value_head_weight_decay: float = field(default=0.0)
    pair_use_t_group_weight: bool = field(default=True)
    save_format: str = field(default="hf")
    save_interval: Optional[int] = field(default=None)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
