from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Sequence

import torch

from openpi.qwenvl.modeling._prompt_templates import PAIR_PROMPT_TEMPLATE
from openpi.qwenvl.modeling._qwen_vl_shared import build_chat_messages
from openpi.qwenvl.modeling._qwen_vl_shared import collate_padded_qwenvl_batch
from openpi.qwenvl.modeling._qwen_vl_shared import get_qwen25_rope_index
from openpi.qwenvl.modeling._qwen_vl_shared import load_qwenvl_data_args
from openpi.qwenvl.modeling._qwen_vl_shared import load_qwenvl_processor_from_path
from openpi.qwenvl.modeling._qwen_vl_shared import move_batch_to_device
from openpi.qwenvl.train.argument import DataArguments
from openpi.qwenvl.train.pair_model import load_pair_model
def load_pair_processor_from_path(model_name_or_path: str | Path, data_args: DataArguments | None = None):
    return load_qwenvl_processor_from_path(model_name_or_path, data_args=data_args)


def load_pair_inference_components(
    checkpoint_dir: str | Path,
    *,
    model_name_or_path: str | None = None,
    attn_implementation: str = "flash_attention_2",
    bf16: bool = True,
    device: str | torch.device | None = None,
    data_args: DataArguments | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    resolved_data_args = load_qwenvl_data_args(checkpoint_path, data_args=data_args)
    model, model_meta = load_pair_model(
        checkpoint_dir=checkpoint_path,
        model_name_or_path=model_name_or_path,
        attn_implementation=attn_implementation,
        bf16=bf16 and torch.cuda.is_available(),
        map_location="cpu",
    )
    processor = load_qwenvl_processor_from_path(
        checkpoint_path,
        data_args=resolved_data_args,
        attn_implementation=attn_implementation,
    )
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target_device)
    model.eval()
    return {
        "checkpoint_dir": checkpoint_path,
        "device": target_device,
        "processor": processor,
        "model": model,
        "model_arch": model.model_type,
        "merge_size": getattr(processor.image_processor, "merge_size", 2),
        "get_rope_index": get_qwen25_rope_index(model.model_type, context="pair inference"),
        "data_args": resolved_data_args,
        "model_meta": model_meta,
    }


def build_pair_source(
    instruction: str,
    current_images: Sequence[Any],
    history_images: Sequence[Any],
    *,
    delta_label: float = 0.0,
    t_group_weight: float = 1.0,
) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": PAIR_PROMPT_TEMPLATE.format(instruction=instruction)},
            {"from": "gpt", "value": ""},
        ],
        "image": list(current_images) + list(history_images),
        "delta_label": float(delta_label),
        "t_group_weight": float(t_group_weight),
    }


def prepare_pair_instance_from_source(
    source: dict[str, Any],
    *,
    processor,
    get_rope_index,
    merge_size: int,
) -> dict[str, Any]:
    messages = build_chat_messages(source)
    data_dict = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if isinstance(data_dict.get("input_ids"), list):
        data_dict["input_ids"] = torch.tensor(data_dict["input_ids"]).unsqueeze(0)
    data_dict["labels"] = torch.full_like(data_dict["input_ids"], -100)
    data_dict["delta_label"] = float(source["delta_label"])
    data_dict["t_group_weight"] = float(source.get("t_group_weight", 1.0))

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


def build_pair_batch(components: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
    instances = [
        prepare_pair_instance_from_source(
            example,
            processor=components["processor"],
            get_rope_index=components["get_rope_index"],
            merge_size=components["merge_size"],
        )
        for example in examples
    ]
    return collate_padded_qwenvl_batch(
        components["processor"].tokenizer,
        instances,
        extra_tensor_fields=(
            ("delta_label", "delta_labels", torch.float32),
            ("t_group_weight", "t_group_weights", torch.float32),
        ),
    )


def collate_pair_batch(tokenizer, instances: list[dict[str, Any]]) -> dict[str, Any]:
    return collate_padded_qwenvl_batch(
        tokenizer,
        instances,
        extra_tensor_fields=(
            ("delta_label", "delta_labels", torch.float32),
            ("t_group_weight", "t_group_weights", torch.float32),
        ),
    )


@torch.inference_mode()
def predict_pair_values_from_batch(components: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    model_inputs = dict(batch)
    model_inputs.pop("delta_labels", None)
    model_inputs.pop("t_group_weights", None)
    with torch.autocast(
        device_type=components["device"].type,
        dtype=torch.bfloat16,
        enabled=components["device"].type == "cuda",
    ):
        return components["model"].sample_values(**model_inputs).detach().float()


@torch.inference_mode()
def predict_pair_values(
    components: dict[str, Any],
    *,
    prompts: str | Sequence[str],
    current_triplets: Sequence[Sequence[Any]],
    history_triplets: Sequence[Sequence[Any]],
    batch_size: int = 8,
) -> torch.Tensor:
    if len(current_triplets) != len(history_triplets):
        raise ValueError("current_triplets and history_triplets must have the same batch size")
    if isinstance(prompts, str):
        prompt_list = [prompts] * len(current_triplets)
    else:
        prompt_list = [str(prompt) for prompt in prompts]
    if len(prompt_list) != len(current_triplets):
        raise ValueError(f"Expected {len(current_triplets)} prompts, got {len(prompt_list)}")

    outputs = []
    for batch_start in range(0, len(current_triplets), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), len(current_triplets))
        examples = [
            build_pair_source(
                instruction=prompt_list[index],
                current_images=current_triplets[index],
                history_images=history_triplets[index],
            )
            for index in range(batch_start, batch_end)
        ]
        batch = build_pair_batch(components, examples)
        batch = move_batch_to_device(batch, components["device"])
        outputs.append(predict_pair_values_from_batch(components, batch).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.zeros((0,), dtype=torch.float32)
