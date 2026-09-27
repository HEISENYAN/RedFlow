from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import transformers

from openpi.qwenvl.data.rope2d import get_rope_index_25
from openpi.qwenvl.modeling._flash_attn_runtime import configure_transformers_attn_runtime
from openpi.qwenvl.train.argument import DataArguments

IGNORE_INDEX = -100
IMAGE_TOKEN_RE = re.compile(r"(<image>|<video>)")
VISUAL_BATCH_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def normalize_qwen25_model_type(model_type: str, *, context: str) -> str:
    normalized = str(model_type).strip().lower()
    if normalized in {"qwen2.5vl", "qwen2_5_vl"}:
        return "qwen2.5vl"
    raise ValueError(f"{context} only supports Qwen2.5-VL, got model_type={model_type!r}")


def get_qwen25_rope_index(model_type: str, *, context: str):
    normalize_qwen25_model_type(model_type, context=context)
    return get_rope_index_25


def load_qwenvl_run_config(model_dir: str | Path) -> dict[str, Any]:
    run_config_path = Path(model_dir).expanduser().resolve() / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing required run_config.json: {run_config_path}")
    with open(run_config_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def load_qwenvl_data_args(model_dir: str | Path, data_args: DataArguments | None = None) -> DataArguments:
    if data_args is not None:
        return data_args
    run_config = load_qwenvl_run_config(model_dir)
    payload = run_config.get("data_args")
    if not isinstance(payload, dict):
        raise ValueError(f"run_config.json is missing a valid data_args payload: {Path(model_dir).expanduser().resolve()}")
    valid_fields = {field.name for field in fields(DataArguments)}
    return DataArguments(**{key: value for key, value in payload.items() if key in valid_fields})


def update_processor_pixels(processor, data_args: DataArguments):
    image_processor = processor.image_processor
    if hasattr(image_processor, "min_pixels") and hasattr(image_processor, "max_pixels"):
        image_processor.min_pixels = data_args.min_pixels
        image_processor.max_pixels = data_args.max_pixels
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        if "height" in image_processor.size and "width" in image_processor.size:
            image_processor.size["height"] = data_args.image_height
            image_processor.size["width"] = data_args.image_width
        else:
            image_processor.size["shortest_edge"] = min(data_args.image_height, data_args.image_width)
            image_processor.size["longest_edge"] = max(data_args.image_height, data_args.image_width)

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        if hasattr(video_processor, "min_pixels") and hasattr(video_processor, "max_pixels"):
            video_processor.min_pixels = data_args.video_min_pixels
            video_processor.max_pixels = data_args.video_max_pixels
        if hasattr(video_processor, "min_frames") and hasattr(video_processor, "max_frames"):
            video_processor.min_frames = data_args.video_min_frames
            video_processor.max_frames = data_args.video_max_frames
        if hasattr(video_processor, "fps"):
            video_processor.fps = data_args.video_fps
        if hasattr(video_processor, "size") and isinstance(video_processor.size, dict):
            video_processor.size["shortest_edge"] = data_args.video_min_pixels
            video_processor.size["longest_edge"] = data_args.video_max_pixels
    return processor


def load_qwenvl_processor_from_path(
    model_name_or_path: str | Path,
    data_args: DataArguments | None = None,
    *,
    attn_implementation: str | None = None,
):
    configure_transformers_attn_runtime(attn_implementation)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_name_or_path))
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
    resolved_data_args = load_qwenvl_data_args(model_name_or_path, data_args=data_args)
    return update_processor_pixels(processor, resolved_data_args)


def _normalize_media(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    if isinstance(items, tuple):
        return list(items)
    return list(items)


def _resolve_video_path(base_path: Path, video: Any) -> str:
    video_str = str(video)
    if os.path.isabs(video_str):
        return video_str
    if not str(base_path):
        return str(Path(video_str).resolve())
    return str((base_path / video_str).resolve())


def _build_media_pools(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_path = Path(str(source.get("data_path", "")))
    image_pool = [{"type": "image", "image": image} for image in _normalize_media(source.get("image"))]
    video_pool = [
        {"type": "video", "video": _resolve_video_path(base_path, video)}
        for video in _normalize_media(source.get("video"))
    ]
    return image_pool, video_pool


def _build_user_content(text: str, image_pool: list[dict[str, Any]], video_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content = []
    for segment in IMAGE_TOKEN_RE.split(text):
        if segment == "<image>":
            if not image_pool:
                raise ValueError("Number of <image> placeholders exceeds provided images.")
            content.append(image_pool.pop(0))
        elif segment == "<video>":
            if not video_pool:
                raise ValueError("Number of <video> placeholders exceeds provided videos.")
            content.append(video_pool.pop(0))
        elif segment.strip():
            content.append({"type": "text", "text": segment.strip()})
    return content


def build_chat_messages(source: dict[str, Any]) -> list[dict[str, Any]]:
    image_pool, video_pool = _build_media_pools(source)
    messages = []
    for turn in source["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text = str(turn["value"])
        if role == "user":
            messages.append(
                {
                    "role": role,
                    "content": _build_user_content(text, image_pool=image_pool, video_pool=video_pool),
                }
            )
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    if image_pool:
        raise ValueError(f"{len(image_pool)} image(s) remain unused in conversation source.")
    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused in conversation source.")
    return messages


def build_generation_messages(source: dict[str, Any]) -> list[dict[str, Any]]:
    return build_chat_messages(
        {
            **source,
            "conversations": [turn for turn in source["conversations"] if turn["from"] == "human"],
        }
    )


def set_attention_length_metadata(instance: dict[str, Any]) -> dict[str, Any]:
    seq_len = int(instance["input_ids"][0].size(0))
    instance["attention_mask"] = [seq_len]
    return instance


def pad_and_cat_position_ids(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded_tensors, dim=1)


def _attach_visual_fields(batch: dict[str, Any], instances: list[dict[str, Any]]) -> dict[str, Any]:
    for key in VISUAL_BATCH_KEYS:
        values = [instance[key] for instance in instances if key in instance]
        if values:
            batch[key] = torch.cat(values, dim=0)
    return batch


def collate_padded_qwenvl_batch(
    tokenizer,
    instances: list[dict[str, Any]],
    *,
    extra_tensor_fields: tuple[tuple[str, str, torch.dtype], ...] = (),
) -> dict[str, Any]:
    input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
    batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),
        "position_ids": pad_and_cat_position_ids([instance["position_ids"] for instance in instances]),
    }
    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.pad_token_id)
    if all("labels" in instance for instance in instances):
        labels = [instance["labels"].squeeze(0) for instance in instances]
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
    batch["input_ids"] = batch["input_ids"][:, : tokenizer.model_max_length]
    batch["attention_mask"] = batch["attention_mask"][:, : tokenizer.model_max_length]
    batch["position_ids"] = batch["position_ids"][:, :, : tokenizer.model_max_length]
    if "labels" in batch:
        batch["labels"] = batch["labels"][:, : tokenizer.model_max_length]
    batch = _attach_visual_fields(batch, instances)
    for source_key, output_key, dtype in extra_tensor_fields:
        batch[output_key] = torch.tensor([instance[source_key] for instance in instances], dtype=dtype)
    return batch


def collate_flattened_qwenvl_batch(
    instances: list[dict[str, Any]],
    *,
    extra_tensor_fields: tuple[tuple[str, str, torch.dtype], ...] = (),
) -> dict[str, Any]:
    batch = {
        "input_ids": torch.cat([instance["input_ids"] for instance in instances], dim=1),
        "labels": torch.cat([instance["labels"] for instance in instances], dim=1),
        "position_ids": torch.cat([instance["position_ids"] for instance in instances], dim=2),
        "attention_mask": torch.cumsum(
            torch.tensor([0] + [item for instance in instances for item in instance["attention_mask"]], dtype=torch.int32),
            dim=0,
            dtype=torch.int32,
        ),
    }
    batch = _attach_visual_fields(batch, instances)
    for source_key, output_key, dtype in extra_tensor_fields:
        batch[output_key] = torch.tensor([instance[source_key] for instance in instances], dtype=dtype)
    return batch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


@dataclass
class QwenVLPaddedCollator:
    tokenizer: transformers.PreTrainedTokenizer
    extra_tensor_fields: tuple[tuple[str, str, torch.dtype], ...] = ()

    def __call__(self, instances):
        return collate_padded_qwenvl_batch(
            self.tokenizer,
            instances,
            extra_tensor_fields=self.extra_tensor_fields,
        )
