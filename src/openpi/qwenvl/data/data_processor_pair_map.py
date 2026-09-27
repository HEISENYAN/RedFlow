from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from openpi.qwenvl.data._module_common import resolve_dataset_paths
from openpi.qwenvl.data._module_common import sync_min_length
from openpi.qwenvl.data._module_common import sync_min_length_list
from openpi.qwenvl.data.data_loader_pair_map import MapPairDataset
from openpi.qwenvl.data.data_processor import parse_camera_names
from openpi.qwenvl.data.data_processor import update_processor_pixels
from openpi.qwenvl.modeling._qwen_vl_shared import get_qwen25_rope_index
from openpi.qwenvl.modeling._qwen_vl_shared import normalize_qwen25_model_type
from openpi.qwenvl.modeling.pair_inference import build_pair_batch
from openpi.qwenvl.modeling.pair_inference import build_pair_source
from openpi.qwenvl.modeling.pair_inference import collate_pair_batch
from openpi.qwenvl.modeling.pair_inference import prepare_pair_instance_from_source


@dataclass
class PairDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        return collate_pair_batch(self.tokenizer, instances)


@dataclass
class FlattenedPairDataCollator(PairDataCollator):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = [item for instance in instances for item in instance["attention_mask"]]
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        batch = {
            "input_ids": torch.cat(input_ids, dim=1),
            "labels": torch.cat(labels, dim=1),
            "position_ids": torch.cat(position_ids, dim=2),
            "attention_mask": cumsum_seq_lens,
            "delta_labels": torch.tensor([instance["delta_label"] for instance in instances], dtype=torch.float32),
            "t_group_weights": torch.tensor(
                [instance.get("t_group_weight", 1.0) for instance in instances], dtype=torch.float32
            ),
        }

        if any("pixel_values" in instance for instance in instances):
            batch["pixel_values"] = torch.cat(
                [instance["pixel_values"] for instance in instances if "pixel_values" in instance],
                dim=0,
            )
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        if any("pixel_values_videos" in instance for instance in instances):
            batch["pixel_values_videos"] = torch.cat(
                [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance],
                dim=0,
            )
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance],
                dim=0,
            )
        return batch


class SupervisedPairMapDataset(Dataset):
    def __init__(self, map_dataset, processor, data_args, max_samples=None):
        super().__init__()
        self.map_dataset = map_dataset
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = self.processor.tokenizer
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.model_type = normalize_qwen25_model_type(getattr(data_args, "model_type", "qwen2.5vl"), context="pair data module")
        self.get_rope_index = get_qwen25_rope_index(self.model_type, context="pair data module")
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)

    def __len__(self):
        base_len = len(self.map_dataset)
        return min(base_len, self.max_samples) if self.max_samples is not None else base_len

    def __getitem__(self, index):
        return prepare_pair_instance(
            self.map_dataset[int(index)],
            processor=self.processor,
            get_rope_index=self.get_rope_index,
            merge_size=self.merge_size,
        )


def prepare_pair_instance(source: dict[str, Any], processor, get_rope_index, merge_size: int) -> dict[str, Any]:
    data_dict = prepare_pair_instance_from_source(
        source,
        processor=processor,
        get_rope_index=get_rope_index,
        merge_size=merge_size,
    )
    seq_len = data_dict["input_ids"][0].size(0)
    data_dict["attention_mask"] = [seq_len]
    return data_dict


class PairBatchBuilder:
    def __init__(self, processor, model_type: str):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        normalized_model_type = normalize_qwen25_model_type(model_type, context="pair batch builder")
        self.get_rope_index = get_qwen25_rope_index(normalized_model_type, context="pair batch builder")
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)
        self.collator = PairDataCollator(self.processor.tokenizer)

    def build_batch(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        _ = self.get_rope_index, self.merge_size, self.collator
        sources = [
            build_pair_source(
                instruction=str(example["instruction"]),
                current_images=list(example["current_images"]),
                history_images=list(example["history_images"]),
                delta_label=float(example.get("delta_label", 0.0)),
                t_group_weight=float(example.get("t_group_weight", 1.0)),
            )
            for example in examples
        ]
        components = {
            "processor": self.processor,
            "get_rope_index": self.get_rope_index,
            "merge_size": self.merge_size,
        }
        return build_pair_batch(components, sources)


def _build_one_repo_map_dataset(dataset_dir: str, split: str, data_args, seed_offset: int = 0):
    return MapPairDataset(
        dataset_dir=dataset_dir,
        split=split,
        val_ratio=getattr(data_args, "val_ratio", 0.1),
        seed=getattr(data_args, "seed", 42) + seed_offset,
        camera_names=parse_camera_names(getattr(data_args, "camera_names", None)),
        max_episodes=getattr(data_args, "max_episodes", None),
        pair_add_backward=getattr(data_args, "pair_add_backward", True),
        pair_add_zero_anchor=getattr(data_args, "pair_add_zero_anchor", None),
        pair_target_column=getattr(data_args, "pair_target_column", None),
        pair_prompt_style=getattr(data_args, "pair_prompt_style", "current_history_delta"),
    )


def make_supervised_pair_map_data_module(processor, data_args, model_args=None, return_stats: bool = False):
    data_args.model_type = getattr(data_args, "model_type", "qwen2.5vl")
    resolved_paths = resolve_dataset_paths(data_args.dataset_use)
    train_raw_parts = []
    val_raw_parts = []
    train_local_lens = []
    val_local_lens = []

    for idx, dataset_dir in enumerate(resolved_paths):
        train_part = _build_one_repo_map_dataset(dataset_dir, "train", data_args, seed_offset=idx * 1000)
        val_part = _build_one_repo_map_dataset(dataset_dir, "val", data_args, seed_offset=idx * 1000)
        train_raw_parts.append(train_part)
        val_raw_parts.append(val_part)
        train_local_lens.append(len(train_part))
        val_local_lens.append(len(val_part))

    if len(resolved_paths) == 1:
        train_synced_lens = [sync_min_length(train_local_lens[0])]
        val_synced_lens = [sync_min_length(val_local_lens[0])]
    else:
        train_synced_lens = sync_min_length_list(train_local_lens)
        val_synced_lens = sync_min_length_list(val_local_lens)

    train_parts = []
    val_parts = []
    for idx, _dataset_dir in enumerate(resolved_paths):
        if train_synced_lens[idx] > 0:
            train_parts.append(SupervisedPairMapDataset(train_raw_parts[idx], processor, data_args, max_samples=train_synced_lens[idx]))
        if val_synced_lens[idx] > 0:
            val_parts.append(SupervisedPairMapDataset(val_raw_parts[idx], processor, data_args, max_samples=val_synced_lens[idx]))

    if not train_parts:
        raise ValueError(f"No valid training samples after sync. local={train_local_lens}, synced={train_synced_lens}")
    if not val_parts:
        raise ValueError(f"No valid eval samples after sync. local={val_local_lens}, synced={val_synced_lens}")

    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    eval_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    if getattr(data_args, "data_flatten", False) or getattr(data_args, "data_packing", False):
        data_collator = FlattenedPairDataCollator(processor.tokenizer)
    else:
        data_collator = PairDataCollator(processor.tokenizer)

    out = {"train_dataset": train_dataset, "eval_dataset": eval_dataset, "data_collator": data_collator}
    if not return_stats:
        return out
    return out, {
        "repo_paths": [os.path.abspath(path) for path in resolved_paths],
        "train_lengths_local": train_local_lens,
        "train_lengths_synced": train_synced_lens,
        "val_lengths_local": val_local_lens,
        "val_lengths_synced": val_synced_lens,
        "train_total_samples": len(train_dataset),
        "eval_total_samples": len(eval_dataset),
    }
