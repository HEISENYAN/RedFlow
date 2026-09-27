from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Optional

import torch
import torch.nn as nn

from openpi.qwenvl.modeling._flash_attn_runtime import configure_transformers_attn_runtime

PAIR_MODEL_CONFIG_NAME = "pair_model_config.json"
PAIR_MODEL_WEIGHTS_NAME = "pair_model.bin"


def get_visual_bridge_module(model):
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError(f"Model {type(model).__name__} does not expose a visual tower")
    merger = getattr(visual, "merger", None)
    if merger is None:
        raise ValueError("Could not locate visual merger module under model.visual.merger")
    return merger


def load_backbone(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    bf16: bool = False,
):
    configure_transformers_attn_runtime(attn_implementation)
    from transformers import Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if bf16 else None
    common_kwargs = {
        "cache_dir": cache_dir,
        "attn_implementation": attn_implementation,
        "torch_dtype": dtype,
    }
    if "qwen2.5" not in model_name_or_path.lower():
        raise ValueError(f"Pair training only supports Qwen2.5-VL, got model_name_or_path={model_name_or_path!r}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **common_kwargs)
    model_type = "qwen2.5vl"
    return model, model_type


def set_backbone_trainable(model_args, backbone):
    bridge = get_visual_bridge_module(backbone)
    for _, param in backbone.visual.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_vision)
    for _, param in bridge.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_mlp)
    for _, param in backbone.language_model.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_llm)
    backbone.lm_head.requires_grad = bool(model_args.tune_mm_llm)


def maybe_untie_word_embeddings(backbone):
    if not getattr(backbone.config, "tie_word_embeddings", False):
        return

    embed_tokens = getattr(backbone.model.language_model, "embed_tokens", None)
    if embed_tokens is None:
        raise ValueError("tie_word_embeddings is enabled but embed_tokens could not be located")

    backbone.config.tie_word_embeddings = False
    backbone.lm_head = nn.Linear(backbone.config.hidden_size, backbone.config.vocab_size, bias=False)
    backbone.lm_head.weight.data = embed_tokens.weight.data.clone()


class QwenPairDeltaModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        bf16: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.backbone, self.model_type = load_backbone(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            bf16=bf16,
        )
        maybe_untie_word_embeddings(self.backbone)
        hidden_size = self.backbone.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        self.gradient_checkpointing_enabled = False

    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_enabled = True
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.backbone.gradient_checkpointing_disable()

    def is_gradient_checkpointing_enabled(self):
        return self.gradient_checkpointing_enabled

    @staticmethod
    def _last_valid_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()
        last_idx = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx, :]

    def _forward_backbone(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        position_ids=None,
    ):
        outputs = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        pooled = self._last_valid_token(hidden_states, attention_mask)
        head_dtype = next(self.value_head.parameters()).dtype
        return self.value_head(pooled.to(dtype=head_dtype)).squeeze(-1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        position_ids=None,
        delta_labels=None,
        t_group_weights=None,
        **kwargs,
    ):
        delta_pred = self._forward_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
        )

        loss = None
        value_mse = None
        if delta_labels is not None:
            pred = delta_pred.float()
            target = torch.clamp(delta_labels, -1.0, 1.0).float()
            mse = (pred - target) ** 2
            value_mse = mse.mean()
            if t_group_weights is not None:
                weights = torch.clamp(t_group_weights, min=0.0).float()
                loss = (mse * weights).sum() / torch.clamp(weights.sum(), min=1e-8)
            else:
                loss = value_mse

        return {
            "loss": loss,
            "value_loss": loss,
            "value_mse": value_mse,
            "delta_pred": delta_pred,
        }

    @torch.inference_mode()
    def sample_values(self, **kwargs) -> torch.Tensor:
        return self._forward_backbone(**kwargs)

    @torch.inference_mode()
    def sample_pair_delta(self, **kwargs) -> torch.Tensor:
        return self.sample_values(**kwargs)


def export_pair_model(model: QwenPairDeltaModel, processor, output_dir: str | Path, metadata: Optional[dict[str, Any]] = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    existing_weights = [
        output_path / PAIR_MODEL_WEIGHTS_NAME,
        output_path / "pytorch_model.bin",
        output_path / "model.safetensors",
    ]
    if not any(path.exists() for path in existing_weights):
        cpu_state_dict = {key: value.detach().cpu() for key, value in model_to_save.state_dict().items()}
        torch.save(cpu_state_dict, output_path / PAIR_MODEL_WEIGHTS_NAME)

    processor.save_pretrained(output_path)
    if hasattr(model_to_save.backbone.config, "save_pretrained"):
        model_to_save.backbone.config.save_pretrained(output_path)

    payload = {
        "base_model_name_or_path": model_to_save.model_name_or_path,
        "model_type": model_to_save.model_type,
    }
    if metadata:
        payload.update(metadata)
    with open(output_path / PAIR_MODEL_CONFIG_NAME, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)


def _load_raw_state_dict(weights_path: Path, map_location: str | torch.device = "cpu"):
    if weights_path.suffix == ".safetensors":
        import safetensors.torch

        return safetensors.torch.load_file(str(weights_path), device=str(map_location))
    return torch.load(weights_path, map_location=map_location, weights_only=False)


def find_pair_model_weights(checkpoint_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    candidates = [
        checkpoint_path / PAIR_MODEL_WEIGHTS_NAME,
        checkpoint_path / "model.safetensors",
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "adapter_model.bin",
        checkpoint_path / "adapter_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported pair model weights found under {checkpoint_path}")


def load_pair_model(
    checkpoint_dir: str | Path,
    model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    bf16: bool = False,
    map_location: str | torch.device = "cpu",
):
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / PAIR_MODEL_CONFIG_NAME
    config_payload = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fin:
            config_payload = json.load(fin)

    base_model_name = (
        model_name_or_path
        or config_payload.get("base_model_name_or_path")
        or config_payload.get("model_name_or_path")
    )
    if not base_model_name:
        raise ValueError(
            "Could not determine base model path for pair model loading. "
            "Pass model_name_or_path explicitly or load from an exported final_model directory."
        )

    model = QwenPairDeltaModel(
        model_name_or_path=base_model_name,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
        bf16=bf16,
    )
    state_dict = _load_raw_state_dict(find_pair_model_weights(checkpoint_path), map_location=map_location)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    critical_missing = [key for key in missing_keys if not key.endswith("num_batches_tracked")]
    if critical_missing or unexpected_keys:
        lora_related = [key for key in unexpected_keys if "lora_" in key or "adapter_" in key]
        if lora_related:
            raise RuntimeError(
                "This checkpoint contains LoRA/adapter weights. "
                "The evaluator expects a fully materialized pair model checkpoint."
            )
        raise RuntimeError(
            "Failed to load pair model checkpoint cleanly. "
            f"missing_keys={critical_missing}, unexpected_keys={unexpected_keys}"
        )
    return model, config_payload
