from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

IGNORE_INDEX = -100
DEFAULT_CAMERA_NAMES = ("cam_high",)
IMAGE_TOKEN_RE = re.compile(r"(<image>|<video>)")


def parse_camera_names(camera_names: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if camera_names is None:
        return list(DEFAULT_CAMERA_NAMES)
    if isinstance(camera_names, str):
        return [part.strip() for part in camera_names.split(",") if part.strip()]
    return [str(part).strip() for part in camera_names if str(part).strip()]


def update_processor_pixels(processor, data_args):
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


def _normalize_media(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    if isinstance(items, tuple):
        return list(items)
    return list(items)


def _normalize_images(images: Any) -> list[Any]:
    return _normalize_media(images)


def _normalize_videos(videos: Any) -> list[Any]:
    return _normalize_media(videos)


def _resolve_video_path(base_path: Path, video: Any) -> str:
    video_str = str(video)
    if os.path.isabs(video_str):
        return video_str
    if not str(base_path):
        return str(Path(video_str).resolve())
    return str((base_path / video_str).resolve())


def _build_media_pools(source: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_path = Path(str(source.get("data_path", "")))
    image_pool = [{"type": "image", "image": image} for image in _normalize_images(source.get("image"))]
    video_pool = [
        {"type": "video", "video": _resolve_video_path(base_path, video)}
        for video in _normalize_videos(source.get("video"))
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


def preprocess_qwen_visual(sources, processor, value_tokenizer=None) -> dict[str, Any]:
    if len(sources) != 1:
        raise ValueError(f"Expected exactly one source, got {len(sources)}")

    source = sources[0]
    full_messages = build_chat_messages(source)
    input_result = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = input_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    if value_tokenizer is not None:
        for pos, token_id in enumerate(input_ids[0].tolist()):
            if int(token_id) in value_tokenizer.extra_id_token_ids:
                labels[0, pos] = input_ids[0, pos]

    input_result["input_ids"] = input_ids
    input_result["labels"] = labels
    return input_result
