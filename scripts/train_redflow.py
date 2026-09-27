"""Train Pi0RedFlow with the RedFlow loss (KTO-Hinge + cluster pull) on the
chunked real_rollout dataset, with LoRA fine-tuning and per-epoch checkpoints.

CLI:
  uv run scripts/train_redflow.py pi0_real_redflow \
      --exp_name real_redflow_folding_clothes \
      --data.repo_id /path/to/real_rollout/folding_clothes/lerobot_chunked \
      --norm_stats_path /path/to/norm_stats.json \
      --num_epochs 30 --batch_size 32 --fsdp_devices 8 --wandb_enabled
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import os
import platform

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
from flax.training import common_utils
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_chunked as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# HF datasets back-compat (carried over from train_brrl.py — required by lerobot)
import datasets.features.features as hf_features
if "List" not in hf_features._FEATURE_TYPES:
    hf_features._FEATURE_TYPES["List"] = hf_features.Sequence


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


# W&B project used when online logging is enabled.
WANDB_PROJECT = "pi0-real-rollout-redflow"


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=WANDB_PROJECT)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=WANDB_PROJECT)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader, params_shape):
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(config: _config.TrainConfig, init_rng, mesh, *, resume: bool):
    """Initialize the optimizer, model weights, and sharded training state."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        # Convert frozen (non-trainable) params to bfloat16 — matches train_brrl.
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding
    partial_params = (
        _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
        if config.weight_loader is not None else None
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init, donate_argnums=(1,),
        in_shardings=replicated_sharding, out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,    # (Observation, Actions, dict[str, Array]) — RL fields in dict
):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng, observation, actions, cluster_advantage, pull_target_action, pull_alpha):
        loss_scalar, info = model.compute_loss_redflow(
            rng,
            observation=observation, actions=actions,
            cluster_advantage=cluster_advantage,
            pull_target_action=pull_target_action,
            pull_alpha=pull_alpha,
            adv_temperature=float(config.adv_temperature),
            kto_gamma=float(config.kto_gamma),
            margin_scale=float(config.margin_scale),
            pull_lambda=float(config.pull_lambda),
            train=True,
        )
        return loss_scalar, info

    base_rng = jax.random.fold_in(rng, state.step)
    train_rng = jax.random.fold_in(base_rng, 0)
    observation, actions, rl_batch = batch
    cluster_advantage = jnp.asarray(rl_batch["cluster_advantage"], dtype=jnp.float32)
    pull_target_action = jnp.asarray(rl_batch["pull_target_action"], dtype=jnp.float32)
    pull_alpha = jnp.asarray(rl_batch["pull_alpha"], dtype=jnp.float32)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, cluster_advantage, pull_target_action, pull_alpha,
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    out_info = {"loss": loss, **info}
    return new_state, out_info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if config.is_debug:
        import debugpy
        print("Waiting for debugger on 5678..."); debugpy.listen(("0.0.0.0", 5678)); debugpy.wait_for_client()

    if config.norm_stats_path is None:
        raise ValueError("--norm_stats_path is required (no auto-resolution).")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {config.batch_size} not divisible by device count {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite if not os.getenv("DEBUG_MODE", default=False) == "true" else True,
        resume=config.resume,
        max_to_keep=config.max_to_keep,
    )
    init_wandb(config, resuming=resuming, log_code=False, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_chunked_data_loader(
        config,
        norm_stats_path=config.norm_stats_path,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    dataset_size = len(data_loader)
    steps_per_epoch = max(1, dataset_size // config.batch_size)
    total_steps = config.num_epochs * steps_per_epoch
    logging.info(f"dataset_size={dataset_size} steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, total_steps), initial=start_step, total=total_steps, dynamic_ncols=True)
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)

        if (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            if epoch % config.save_every_n_epochs == 0:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step + 1)
                wandb.log({"epoch": epoch}, step=step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
