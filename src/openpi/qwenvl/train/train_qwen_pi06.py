from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
from dataclasses import asdict
from pathlib import Path

import datasets.features.features as hf_features
import torch
import torch.distributed as dist
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import transformers

from openpi.qwenvl.data.data_module import build_training_data_module
from openpi.qwenvl.modeling._flash_attn_runtime import configure_transformers_attn_runtime
from openpi.qwenvl.modeling._qwen_vl_shared import normalize_qwen25_model_type
from openpi.qwenvl.modeling.pi06_inference import build_value_tokenizer
from openpi.qwenvl.modeling.pi06_inference import load_pi06_generation_model_from_path
from openpi.qwenvl.modeling.pi06_inference import load_pi06_processor_from_path
from openpi.qwenvl.train.argument import DataArguments
from openpi.qwenvl.train.argument import ModelArguments
from openpi.qwenvl.train.argument import TrainingArguments

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).print_help()
    raise SystemExit(0)

TRAINER_IMPORT_ERROR = None
try:
    from transformers import Trainer
    from transformers import TrainerCallback
except Exception as exc:  # pragma: no cover - depends on local binary env
    Trainer = None  # type: ignore[assignment]

    class TrainerCallback:  # type: ignore[no-redef]
        pass

    TRAINER_IMPORT_ERROR = exc

if "List" not in hf_features._FEATURE_TYPES:
    hf_features._FEATURE_TYPES["List"] = hf_features.Sequence


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    for _, param in model.visual.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_vision)
    for _, param in model.visual.merger.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_mlp)
    for _, param in model.language_model.named_parameters():
        param.requires_grad = bool(model_args.tune_mm_llm)
    model.lm_head.requires_grad = bool(model_args.tune_mm_llm)


class StructuredMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str, config_payload: dict):
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "train_metrics.jsonl"
        self.config_path = self.output_dir / "run_config.json"
        self.config_payload = config_payload

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as fout:
            json.dump(self.config_payload, fout, ensure_ascii=True, indent=2, sort_keys=True)
        if self.metrics_path.exists():
            self.metrics_path.unlink()

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if not state.is_world_process_zero or not logs:
            return
        record = {"global_step": int(state.global_step), "epoch": float(state.epoch) if state.epoch is not None else None}
        record.update(logs)
        with open(self.metrics_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_training_summary(trainer: Trainer, output_dir: str, extra: dict):
    if not trainer.is_world_process_zero():
        return
    summary_path = Path(output_dir) / "train_summary.json"
    last_logged = {}
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry or "eval_loss" in entry:
            last_logged = dict(entry)
            break
    payload = {
        "global_step": int(trainer.state.global_step),
        "best_metric": trainer.state.best_metric,
        "last_log": last_logged,
    }
    payload.update(extra)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)


def _copy_run_config_if_present(src_dir: str | Path, dst_dir: str | Path):
    src = Path(src_dir) / "run_config.json"
    if not src.exists():
        return
    dst = Path(dst_dir) / "run_config.json"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def train(attn_implementation="flash_attention_2"):
    configure_transformers_attn_runtime(attn_implementation)
    if TRAINER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import transformers.Trainer. "
            "The current environment cannot initialize the flash-attention runtime required by this build."
        ) from TRAINER_IMPORT_ERROR
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not getattr(data_args, "use_value_tokenizer", False):
        raise ValueError("train_qwen_pi06.py requires --use_value_tokenizer True")

    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.save_interval is not None:
        training_args.save_strategy = "steps"
        training_args.save_steps = int(training_args.save_interval)

    model, model_type = load_pi06_generation_model_from_path(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
        bf16=training_args.bf16,
    )
    data_args.model_type = normalize_qwen25_model_type(model_type, context="pi06 training")
    model.config.use_cache = False

    processor = load_pi06_processor_from_path(
        model_args.model_name_or_path,
        data_args=data_args,
        attn_implementation=attn_implementation,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.model_max_length = training_args.model_max_length
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise ValueError("AutoProcessor did not provide tokenizer for pi06 training")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                del module, _input
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig
        from peft import TaskType
        from peft import get_peft_model

        for param in model.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_prefetch_factor = 2
    training_args.dataloader_pin_memory = True
    training_args.dataloader_persistent_workers = training_args.dataloader_num_workers > 0

    value_tokenizer = build_value_tokenizer(model_args.model_name_or_path, data_args=data_args)
    data_module, stats = build_training_data_module(
        processor,
        data_args=data_args,
        model_args=model_args,
        value_tokenizer=value_tokenizer,
        return_stats=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[
            StructuredMetricsCallback(
                output_dir=training_args.output_dir,
                config_payload={
                    "model_args": asdict(model_args),
                    "data_args": asdict(data_args),
                    "training_args": training_args.to_dict(),
                },
            )
        ],
        **data_module,
    )

    ckpt_list = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    try:
        if ckpt_list:
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception:
        logging.exception("Pi06 map training failed on this rank")
        if dist.is_initialized():
            dist.abort()
        raise

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_dir)
    processor.save_pretrained(final_model_dir)
    _copy_run_config_if_present(training_args.output_dir, final_model_dir)
    write_training_summary(
        trainer,
        training_args.output_dir,
        extra={
            "final_model_dir": final_model_dir,
            "repo_paths": stats["repo_paths"],
            "train_total_samples": stats["train_total_samples"],
            "eval_total_samples": stats["eval_total_samples"],
        },
    )


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
