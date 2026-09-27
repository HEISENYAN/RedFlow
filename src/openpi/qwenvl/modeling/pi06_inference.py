from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

import torch
from PIL import Image
import numpy as np

from openpi.qwenvl.modeling._prompt_templates import VALUE_PROMPT_TEMPLATE
from openpi.qwenvl.modeling._flash_attn_runtime import configure_transformers_attn_runtime
from openpi.qwenvl.modeling._qwen_vl_shared import normalize_qwen25_model_type
from openpi.qwenvl.data._lerobot_dataset import resize_pil_rgb_image
from openpi.qwenvl.data.data_processor import build_chat_messages
from openpi.qwenvl.data.data_processor import build_generation_messages
from openpi.qwenvl.data.data_processor import update_processor_pixels
from openpi.qwenvl.data.rope2d import get_rope_index_25
from openpi.qwenvl.train.argument import DataArguments
from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def load_pi06_run_config(model_dir: str | Path) -> dict[str, Any]:
    candidate = Path(model_dir) / "run_config.json"
    if not candidate.exists():
        return {}
    with open(candidate, "r", encoding="utf-8") as fin:
        return json.load(fin)


def assert_strict_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_dir does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(f"checkpoint_dir must be a directory: {checkpoint_path}")
    for filename in REQUIRED_CHECKPOINT_FILES:
        if not (checkpoint_path / filename).exists():
            raise FileNotFoundError(f"checkpoint_dir is missing required file {filename}: {checkpoint_path}")
    if not any(
        (checkpoint_path / name).exists()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "adapter_model.bin",
            "adapter_model.safetensors",
        )
    ):
        raise FileNotFoundError(f"checkpoint_dir does not contain model weights: {checkpoint_path}")
    return checkpoint_path


def build_value_tokenizer(llm_path: str | Path, data_args: DataArguments | None = None) -> ValueTokenizer:
    args = data_args or DataArguments()
    return ValueTokenizer(
        llm_path=str(llm_path),
        bins=int(args.value_tokenizer_bins),
        min_value=float(args.value_tokenizer_min),
        max_value=float(args.value_tokenizer_max),
    )


def load_pi06_processor_from_path(
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
    return update_processor_pixels(processor, data_args or DataArguments())


def load_pi06_processor(
    checkpoint_dir: str | Path,
    data_args: DataArguments | None = None,
    *,
    attn_implementation: str | None = None,
):
    return load_pi06_processor_from_path(
        assert_strict_checkpoint_dir(checkpoint_dir),
        data_args=data_args,
        attn_implementation=attn_implementation,
    )


def load_pi06_generation_model_from_path(
    model_name_or_path: str | Path,
    *,
    attn_implementation: str | None = None,
    bf16: bool = False,
):
    configure_transformers_attn_runtime(attn_implementation)
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers import AutoConfig

    model_path = str(model_name_or_path)
    config = AutoConfig.from_pretrained(model_path)
    model_type = normalize_qwen25_model_type(str(getattr(config, "model_type", "")), context="pi06 model loading")
    kwargs: dict[str, Any] = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if bf16:
        kwargs["torch_dtype"] = torch.bfloat16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    return model, model_type


def load_pi06_generation_model(
    checkpoint_dir: str | Path,
    *,
    attn_implementation: str | None = None,
    bf16: bool = False,
):
    checkpoint_path = assert_strict_checkpoint_dir(checkpoint_dir)
    return load_pi06_generation_model_from_path(
        checkpoint_path,
        attn_implementation=attn_implementation,
        bf16=bf16,
    )


def _get_rope_index_fn(model_arch: str):
    normalize_qwen25_model_type(model_arch, context="pi06 rope selection")
    return get_rope_index_25


def load_pi06_inference_components(
    checkpoint_dir: str | Path,
    *,
    attn_implementation: str = "flash_attention_2",
    bf16: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    checkpoint_path = assert_strict_checkpoint_dir(checkpoint_dir)
    processor = load_pi06_processor(checkpoint_path, attn_implementation=attn_implementation)
    model, model_arch = load_pi06_generation_model(
        checkpoint_path,
        attn_implementation=attn_implementation,
        bf16=bf16 and torch.cuda.is_available(),
    )
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    tokenizer_ref = checkpoint_path
    value_tokenizer = build_value_tokenizer(tokenizer_ref)
    if getattr(model, "generation_config", None) is not None and processor.tokenizer is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    return {
        "checkpoint_dir": checkpoint_path,
        "device": target_device,
        "processor": processor,
        "model": model,
        "model_arch": model_arch,
        "merge_size": getattr(processor.image_processor, "merge_size", 2),
        "get_rope_index": _get_rope_index_fn(model_arch),
        "value_tokenizer": value_tokenizer,
    }


def _pad_and_cat_position_ids(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded_tensors, dim=1)


def _normalize_prompt_batch(prompts: str | Sequence[str], batch_size: int) -> list[str]:
    if isinstance(prompts, str):
        return [prompts] * int(batch_size)
    prompt_list = [str(prompt) for prompt in prompts]
    if len(prompt_list) != int(batch_size):
        raise ValueError(f"Expected {batch_size} prompts, got {len(prompt_list)}")
    return prompt_list


def _to_pil_rgb(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"pi06 inference expects pre-normalized PIL images from the LeRobot data pipeline, got {type(image)!r}"
        )
    return resize_pil_rgb_image(image)


def _as_view_images(frame_views: Sequence[Any]) -> list[Image.Image]:
    if len(frame_views) < 1:
        raise ValueError("Expected at least one camera view.")
    return [_to_pil_rgb(frame) for frame in frame_views]


def build_pi06_prompt_messages(prompt: str, frame_views: Sequence[Any]) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": VALUE_PROMPT_TEMPLATE.format(instruction=prompt)},
            {"from": "gpt", "value": ""},
        ],
        "image": _as_view_images(frame_views),
    }


def prepare_pi06_instance_from_source(
    source: dict[str, Any],
    *,
    processor,
    get_rope_index,
    merge_size: int,
    value_tokenizer,
    generation_mode: bool,
) -> dict[str, Any]:
    prompt_messages = build_generation_messages(source)
    full_messages = build_chat_messages(source)
    messages = prompt_messages if generation_mode else full_messages
    data_dict = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=bool(generation_mode),
        return_dict=True,
        return_tensors="pt",
    )
    if isinstance(data_dict.get("input_ids"), list):
        data_dict["input_ids"] = torch.tensor(data_dict["input_ids"]).unsqueeze(0)

    if not generation_mode:
        labels = torch.full_like(data_dict["input_ids"], -100)
        if value_tokenizer is None:
            raise ValueError("value_tokenizer is required for pi06 training instance preparation")
        for pos, token_id in enumerate(data_dict["input_ids"][0].tolist()):
            if int(token_id) in value_tokenizer.extra_id_token_ids:
                labels[0, pos] = data_dict["input_ids"][0, pos]
        data_dict["labels"] = labels

    grid_thw = data_dict.get("image_grid_thw")
    if grid_thw is not None and not isinstance(grid_thw, (list, tuple)):
        grid_thw = [grid_thw]
    video_grid_thw = data_dict.get("video_grid_thw")
    if video_grid_thw is not None and not isinstance(video_grid_thw, (list, tuple)):
        video_grid_thw = [video_grid_thw]
    if video_grid_thw is not None:
        second_per_grid_ts = torch.full(
            (len(video_grid_thw),),
            processor.video_processor.temporal_patch_size / processor.video_processor.fps,
            dtype=torch.float32,
            device=data_dict["input_ids"].device,
        )
    else:
        second_per_grid_ts = None

    position_ids, _ = get_rope_index(
        merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
        second_per_grid_ts=second_per_grid_ts,
    )
    data_dict["position_ids"] = position_ids
    return data_dict


def build_pi06_instance(components: dict[str, Any], prompt: str, frame_views: Sequence[Any]) -> dict[str, Any]:
    processor = components["processor"]
    source = build_pi06_prompt_messages(prompt, frame_views)
    instance = prepare_pi06_instance_from_source(
        source,
        processor=processor,
        get_rope_index=components["get_rope_index"],
        merge_size=components["merge_size"],
        value_tokenizer=components["value_tokenizer"],
        generation_mode=True,
    )
    if "attention_mask" not in instance:
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("processor.tokenizer.pad_token_id must be set")
        instance["attention_mask"] = instance["input_ids"].ne(pad_token_id).long()
    return instance


def collate_pi06_batch(components: dict[str, Any], instances: list[dict[str, Any]]) -> dict[str, Any]:
    processor = components["processor"]
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("processor.tokenizer.pad_token_id must be set")
    input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
    attention_masks = [instance["attention_mask"].squeeze(0) for instance in instances]
    batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0),
        "position_ids": _pad_and_cat_position_ids([instance["position_ids"] for instance in instances]),
    }
    if any("token_type_ids" in instance for instance in instances):
        token_type_ids = [instance["token_type_ids"].squeeze(0) for instance in instances]
        batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        values = [instance[key] for instance in instances if key in instance]
        if values:
            batch[key] = torch.cat(values, dim=0)
    return batch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


@torch.inference_mode()
def predict_pi06_values_from_batch(components: dict[str, Any], batch: dict[str, Any]) -> np.ndarray:
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        raise ValueError("attention_mask is required for pi06 inference")
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape={tuple(attention_mask.shape)}")
    last_token_indices = attention_mask.sum(dim=1) - 1
    if torch.any(last_token_indices < 0):
        raise ValueError(f"Found empty prompt rows in attention_mask: {last_token_indices.tolist()}")

    model_inputs = {
        key: value
        for key, value in batch.items()
        if key in {
            "input_ids",
            "attention_mask",
            "position_ids",
            "token_type_ids",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        }
    }
    model = components["model"]
    with torch.autocast(
        device_type=components["device"].type,
        dtype=torch.bfloat16,
        enabled=components["device"].type == "cuda",
    ):
        outputs = model(**model_inputs, return_dict=True)
    logits = outputs.logits
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    next_token_logits = logits[batch_indices, last_token_indices]

    value_token_ids = torch.as_tensor(
        components["value_tokenizer"].extra_id_token_ids,
        device=next_token_logits.device,
        dtype=torch.long,
    )
    value_logits = next_token_logits.index_select(dim=-1, index=value_token_ids)
    selected_bin_indices = value_logits.argmax(dim=-1)
    selected_token_ids = value_token_ids.index_select(dim=0, index=selected_bin_indices)
    decoded = components["value_tokenizer"].decode_token_ids_to_values(selected_token_ids.detach().cpu().numpy())
    return np.asarray(decoded, dtype=np.float32)


@torch.inference_mode()
def predict_pi06_values(
    components: dict[str, Any],
    prompts: str | Sequence[str],
    frame_views_batch: Sequence[Sequence[Any]],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    prompt_list = _normalize_prompt_batch(prompts, len(frame_views_batch))
    outputs = []
    for batch_start in range(0, len(frame_views_batch), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), len(frame_views_batch))
        instances = [
            build_pi06_instance(components, prompt_list[index], frame_views_batch[index])
            for index in range(batch_start, batch_end)
        ]
        batch = collate_pi06_batch(components, instances)
        batch = move_batch_to_device(batch, components["device"])
        outputs.append(torch.from_numpy(predict_pi06_values_from_batch(components, batch)))
    return torch.cat(outputs, dim=0).numpy().astype(np.float32) if outputs else np.zeros((0,), dtype=np.float32)
