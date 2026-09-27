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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from openpi.qwenvl.data.data_processor_pair_map import make_supervised_pair_map_data_module
from openpi.qwenvl.modeling._flash_attn_runtime import configure_transformers_attn_runtime
from openpi.qwenvl.modeling._qwen_vl_shared import load_qwenvl_processor_from_path
from openpi.qwenvl.modeling._qwen_vl_shared import normalize_qwen25_model_type
from openpi.qwenvl.train.argument import DataArguments
from openpi.qwenvl.train.argument import ModelArguments
from openpi.qwenvl.train.argument import TrainingArguments
from openpi.qwenvl.train.pair_model import QwenPairDeltaModel
from openpi.qwenvl.train.pair_model import export_pair_model
from openpi.qwenvl.train.pair_model import set_backbone_trainable

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
    trainer.save_model(output_dir)


class EpochAwareDataLoader(DataLoader):
    def set_epoch(self, epoch: int):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class PairRegressionTrainer(Trainer if Trainer is not None else object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_samples_local = 0
        self._latest_value_loss = None
        self._latest_value_mse = None

    def _build_loader(self, dataset, batch_size: int, shuffle: bool, drop_last: bool):
        sampler = None
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                drop_last=drop_last,
            )

        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": sampler is None and shuffle,
            "sampler": sampler,
            "collate_fn": self.data_collator,
            "drop_last": drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": True,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return EpochAwareDataLoader(**kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train_dataset")
        return self._build_loader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            drop_last=bool(self.args.dataloader_drop_last),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer requires an eval_dataset")
        return self._build_loader(
            dataset=eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            drop_last=bool(self.args.dataloader_drop_last),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        delta_labels = inputs.pop("delta_labels")
        self._seen_samples_local += int(delta_labels.shape[0])
        t_group_weights = inputs.pop("t_group_weights", None)
        if not getattr(self.args, "pair_use_t_group_weight", True):
            t_group_weights = None
        outputs = model(**inputs, delta_labels=delta_labels, t_group_weights=t_group_weights)
        loss = outputs["loss"]
        if outputs.get("value_loss") is not None:
            self._latest_value_loss = float(outputs["value_loss"].detach().float().item())
        if outputs.get("value_mse") is not None:
            self._latest_value_mse = float(outputs["value_mse"].detach().float().item())
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if self._latest_value_loss is not None and "value_loss" not in logs:
            logs["value_loss"] = float(self._latest_value_loss)
        if self._latest_value_mse is not None and "value_mse" not in logs:
            logs["value_mse"] = float(self._latest_value_mse)
        logs["samples_seen_local"] = float(self._seen_samples_local)
        if dist.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            tensor = torch.tensor([float(self._seen_samples_local)], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if self.is_world_process_zero():
                logs["samples_seen_global"] = float(tensor.item())
        super().log(logs, *args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        if getattr(self.args, "value_head_lr", None) is None:
            return super().create_optimizer()

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        head_params_decay = []
        head_params_nodecay = []
        base_params_decay = []
        base_params_nodecay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            in_head = name.startswith("value_head")
            is_no_decay = any(token in name for token in no_decay)
            if in_head and not is_no_decay:
                head_params_decay.append(param)
            elif in_head and is_no_decay:
                head_params_nodecay.append(param)
            elif not in_head and not is_no_decay:
                base_params_decay.append(param)
            else:
                base_params_nodecay.append(param)

        optimizer_grouped_parameters = [
            {"params": base_params_decay, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": base_params_nodecay, "weight_decay": 0.0, "lr": self.args.learning_rate},
            {
                "params": head_params_decay,
                "weight_decay": getattr(self.args, "value_head_weight_decay", 0.0),
                "lr": self.args.value_head_lr,
            },
            {"params": head_params_nodecay, "weight_decay": 0.0, "lr": self.args.value_head_lr},
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer


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


def _final_samples_seen_global(local_count: int) -> float:
    if not dist.is_initialized():
        return float(local_count)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(local_count)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def write_training_summary(trainer: PairRegressionTrainer, output_dir: str, extra: dict):
    if not trainer.is_world_process_zero():
        return
    summary_path = Path(output_dir) / "train_summary.json"
    last_logged = {}
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry or "eval_loss" in entry or "value_loss" in entry:
            last_logged = dict(entry)
            break
    payload = {
        "global_step": int(trainer.state.global_step),
        "best_metric": trainer.state.best_metric,
        "samples_seen_local": float(trainer._seen_samples_local),
        "samples_seen_global": _final_samples_seen_global(trainer._seen_samples_local),
        "last_log": last_logged,
    }
    payload.update(extra)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)


def train(attn_implementation="flash_attention_2"):
    configure_transformers_attn_runtime(attn_implementation)
    if TRAINER_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import transformers.Trainer. "
            "The current environment cannot initialize the flash-attention runtime required by this build."
        ) from TRAINER_IMPORT_ERROR
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.save_interval is not None:
        training_args.save_strategy = "steps"
        training_args.save_steps = int(training_args.save_interval)

    model = QwenPairDeltaModel(
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        bf16=training_args.bf16,
    )
    data_args.model_type = normalize_qwen25_model_type(model.model_type, context="pair training")
    model.backbone.config.use_cache = False

    processor = load_qwenvl_processor_from_path(
        model_args.model_name_or_path,
        data_args=data_args,
        attn_implementation=attn_implementation,
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.model_max_length = training_args.model_max_length
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise ValueError("AutoProcessor did not provide tokenizer for pair training")

    if training_args.gradient_checkpointing:
        if hasattr(model.backbone, "enable_input_require_grads"):
            model.backbone.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                del module, _input
                output.requires_grad_(True)
            model.backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig
        from peft import TaskType
        from peft import get_peft_model

        for param in model.backbone.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model.backbone = get_peft_model(model.backbone, lora_config)
        for param in model.value_head.parameters():
            param.requires_grad = True
    else:
        set_backbone_trainable(model_args, model.backbone)
        for param in model.value_head.parameters():
            param.requires_grad = True

    training_args.dataloader_drop_last = True
    training_args.dataloader_pin_memory = True
    training_args.dataloader_persistent_workers = training_args.dataloader_num_workers > 0
    if training_args.dataloader_num_workers > 0:
        training_args.dataloader_prefetch_factor = 2

    data_module, stats = make_supervised_pair_map_data_module(processor, data_args=data_args, model_args=model_args, return_stats=True)

    trainer = PairRegressionTrainer(
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
        logging.exception("Pair map training failed on this rank")
        if dist.is_initialized():
            dist.abort()
        raise

    trainer.save_state()
    model.backbone.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    final_model_dir = os.path.join(training_args.output_dir, "final_model")
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=final_model_dir)
    processor.save_pretrained(final_model_dir)
    export_pair_model(
        model=trainer.model,
        processor=processor,
        output_dir=final_model_dir,
        metadata={
            "save_format": training_args.save_format,
            "run_name": training_args.run_name,
            "output_dir": training_args.output_dir,
            "lora_enable": bool(training_args.lora_enable),
        },
    )
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
