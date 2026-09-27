from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import torch
import transformers
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset

from openpi.qwenvl.data._module_common import resolve_dataset_paths
from openpi.qwenvl.data._module_common import sync_min_length
from openpi.qwenvl.data._module_common import sync_min_length_list
from openpi.qwenvl.data.data_loader_pi06_map import MapPi06Dataset
from openpi.qwenvl.data.data_processor import IGNORE_INDEX
from openpi.qwenvl.data.data_processor import parse_camera_names
from openpi.qwenvl.data.data_processor import update_processor_pixels
from openpi.qwenvl.modeling._qwen_vl_shared import get_qwen25_rope_index
from openpi.qwenvl.modeling._qwen_vl_shared import normalize_qwen25_model_type
from openpi.qwenvl.modeling.pi06_inference import prepare_pi06_instance_from_source


def pad_and_cat(tensor_list: list[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensors.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded_tensors, dim=1)


@dataclass
class Pi06DataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
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


@dataclass
class FlattenedPi06DataCollator(Pi06DataCollator):
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


class SupervisedPi06MapDataset(Dataset):
    def __init__(self, map_dataset, processor, data_args, value_tokenizer, max_samples=None):
        super().__init__()
        self.map_dataset = map_dataset
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = self.processor.tokenizer
        self.value_tokenizer = value_tokenizer
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.model_type = normalize_qwen25_model_type(getattr(data_args, "model_type", "qwen2.5vl"), context="pi06 data module")
        self.get_rope_index = get_qwen25_rope_index(self.model_type, context="pi06 data module")
        self.merge_size = getattr(self.processor.image_processor, "merge_size", 2)

    def __len__(self):
        base_len = len(self.map_dataset)
        return min(base_len, self.max_samples) if self.max_samples is not None else base_len

    def __getitem__(self, index):
        return prepare_pi06_instance(
            self.map_dataset[int(index)],
            processor=self.processor,
            value_tokenizer=self.value_tokenizer,
            get_rope_index=self.get_rope_index,
            merge_size=self.merge_size,
        )


def prepare_pi06_instance(source: dict[str, Any], processor, value_tokenizer, get_rope_index, merge_size: int):
    data_dict = prepare_pi06_instance_from_source(
        source,
        processor=processor,
        get_rope_index=get_rope_index,
        merge_size=merge_size,
        value_tokenizer=value_tokenizer,
        generation_mode=False,
    )
    seq_len = data_dict["input_ids"][0].size(0)
    data_dict["attention_mask"] = [seq_len]
    return data_dict


def _build_one_repo_map_dataset(dataset_dir: str, split: str, data_args, value_tokenizer, seed_offset: int = 0):
    return MapPi06Dataset(
        dataset_dir=dataset_dir,
        split=split,
        val_ratio=getattr(data_args, "val_ratio", 0.1),
        seed=getattr(data_args, "seed", 42) + seed_offset,
        camera_names=parse_camera_names(getattr(data_args, "camera_names", None)),
        value_tokenizer=value_tokenizer,
        max_episodes=getattr(data_args, "max_episodes", None),
        value_source=getattr(data_args, "value_source", "dataset"),
        value_column=getattr(data_args, "value_column", "value"),
    )


def make_supervised_pi06_map_data_module(
    processor,
    data_args,
    model_args=None,
    value_tokenizer=None,
    return_stats: bool = False,
):
    data_args.model_type = getattr(data_args, "model_type", "qwen2.5vl")

    if value_tokenizer is None:
        if not getattr(data_args, "use_value_tokenizer", False):
            raise ValueError("pi06 map training requires use_value_tokenizer=True")
        if model_args is None:
            raise ValueError("model_args is required when value_tokenizer is not provided")
        from openpi.qwenvl.utils.value_tokenizer import ValueTokenizer

        value_tokenizer = ValueTokenizer(
            llm_path=model_args.model_name_or_path,
            bins=data_args.value_tokenizer_bins,
            min_value=data_args.value_tokenizer_min,
            max_value=data_args.value_tokenizer_max,
        )

    resolved_paths = resolve_dataset_paths(data_args.dataset_use)
    train_raw_parts = []
    val_raw_parts = []
    train_local_lens = []
    val_local_lens = []

    for idx, dataset_dir in enumerate(resolved_paths):
        train_part = _build_one_repo_map_dataset(dataset_dir, "train", data_args, value_tokenizer, seed_offset=idx * 1000)
        val_part = _build_one_repo_map_dataset(dataset_dir, "val", data_args, value_tokenizer, seed_offset=idx * 1000)
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
            train_parts.append(
                SupervisedPi06MapDataset(
                    train_raw_parts[idx],
                    processor,
                    data_args,
                    value_tokenizer=value_tokenizer,
                    max_samples=train_synced_lens[idx],
                )
            )
        if val_synced_lens[idx] > 0:
            val_parts.append(
                SupervisedPi06MapDataset(
                    val_raw_parts[idx],
                    processor,
                    data_args,
                    value_tokenizer=value_tokenizer,
                    max_samples=val_synced_lens[idx],
                )
            )

    if not train_parts:
        raise ValueError(f"No valid training samples after sync. local={train_local_lens}, synced={train_synced_lens}")
    if not val_parts:
        raise ValueError(f"No valid eval samples after sync. local={val_local_lens}, synced={val_synced_lens}")

    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    eval_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    if getattr(data_args, "data_flatten", False) or getattr(data_args, "data_packing", False):
        data_collator = FlattenedPi06DataCollator(processor.tokenizer)
    else:
        data_collator = Pi06DataCollator(processor.tokenizer)

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
