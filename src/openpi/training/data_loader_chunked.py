"""Chunked LeRobot v2.1 dataloader for RedFlow offline RL fine-tuning.

Reads one task's `lerobot_chunked/` directory and surfaces every RL metadata
field (cluster_advantage, pull_target_action, pull_alpha, is_success,
progress_value, cluster_label) alongside the standard observation/action
fields.

Norm stats are loaded from a user-supplied path (`--norm_stats_path`) — no
config-name auto-resolution.

Pipeline order (matches `transform_dataset` in data_loader.py):

  1. PromptFromLeRobotTask (pre-pipeline, adds `prompt` field)
  2. repack_transforms (renames raw lerobot keys -> model keys; passthrough RL fields)
  3. data_transforms.inputs (custom: ReshapeChunkedActions ->
     NormalizeAuxActions, then existing AlohaInputs)
  4. Normalize (uses user-supplied norm_stats for actions/state)
  5. model_transforms (tokenize prompt, resize images, etc.)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import SupportsIndex, TypeVar

import jax
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.models.model as _model
from openpi.shared import normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
logger = logging.getLogger(__name__)

ACTION_HORIZON = 50
RAW_ACTION_DIM = 14   # ALOHA 7+7
RAW_ACTION_FLAT = ACTION_HORIZON * RAW_ACTION_DIM  # 700


# ----- Transforms -------------------------------------------------------------

class ReshapeChunkedActions:
    """Reshape flat `[700]` arrays into `[50, 14]` for the chunked dataset's
    actions / pull_target_action fields.

    Runs after repack_transforms, so it reads the **renamed** key `actions`
    (mapped from raw `action`). The `pull_target_action` key is
    passthrough/identity-renamed by repack.
    """

    @staticmethod
    def _reshape_if_flat(value):
        arr = np.asarray(value, dtype=np.float32)
        # Some lerobot rows produce a singleton leading dim (delta_timestamps);
        # squeeze it before reshape.
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[-1] == RAW_ACTION_FLAT:
            arr = arr.reshape(RAW_ACTION_FLAT)
        if arr.shape == (RAW_ACTION_FLAT,):
            return arr.reshape(ACTION_HORIZON, RAW_ACTION_DIM)
        if arr.shape[-1] == RAW_ACTION_FLAT and arr.ndim >= 2:
            return arr.reshape(*arr.shape[:-1], ACTION_HORIZON, RAW_ACTION_DIM)
        return arr

    def __call__(self, row: dict) -> dict:
        out = dict(row)
        if "actions" in out:
            out["actions"] = self._reshape_if_flat(out["actions"])
        if "pull_target_action" in out:
            out["pull_target_action"] = self._reshape_if_flat(out["pull_target_action"])
        return out


class NormalizeAuxActions:
    """Apply the same normalization to pull_target_action that the standard
    Normalize transform applies to `actions`, using the user-supplied norm
    stats.

    The standard `Normalize` transform only touches keys present in
    `data_config.norm_stats` (typically `actions` and `state`). The auxiliary
    field isn't in norm_stats, so we normalize it manually here using
    the same `actions` mean/std.
    """

    def __init__(
        self,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        *,
        model_action_dim: int = 32,
    ):
        self._action_mean = np.asarray(action_mean, dtype=np.float32)
        self._action_std = np.asarray(action_std, dtype=np.float32)
        self._model_action_dim = model_action_dim

    def _normalize_and_pad(self, x):
        """Normalize with action stats, then right-pad last dim to model_action_dim."""
        x = np.asarray(x, dtype=np.float32)
        normed = (x - self._action_mean) / (self._action_std + 1e-6)
        if normed.shape[-1] < self._model_action_dim:
            pad_width = [(0, 0)] * normed.ndim
            pad_width[-1] = (0, self._model_action_dim - normed.shape[-1])
            normed = np.pad(normed, pad_width, mode="constant", constant_values=0.0)
        return normed.astype(np.float32)

    def __call__(self, row: dict) -> dict:
        out = dict(row)
        if "pull_target_action" in out:
            out["pull_target_action"] = self._normalize_and_pad(out["pull_target_action"])
        return out


# RL passthrough keys — preserved through every downstream transform so they
# survive AlohaInputs (builds a fresh dict), Normalize, and model_transforms.
_RL_PASSTHROUGH = (
    "cluster_advantage", "cluster_label", "pull_target_action",
    "pull_alpha", "is_success", "progress_value",
)


class WithExtras:
    """Wrap a DataTransformFn so a fixed set of extra keys survives the call.

    Saves the listed keys from the input dict before the inner transform,
    then re-attaches them on the way out. Re-attach overwrites any keys the
    inner transform might have produced under the same name.
    """

    def __init__(self, inner, extra_keys=_RL_PASSTHROUGH):
        self._inner = inner
        self._extra_keys = tuple(extra_keys)

    def __call__(self, data: dict) -> dict:
        extras = {k: data[k] for k in self._extra_keys if k in data}
        out = self._inner(data)
        if not isinstance(out, dict):
            return out
        for k, v in extras.items():
            out[k] = v
        return out


# ----- Filtered dataset -------------------------------------------------------

class FilteredLeRobotDataset:
    """Wraps a LeRobotDataset, dropping rows where `cluster_label == -2`
    (post-success chunks). Implements the Dataset protocol used downstream.
    """

    def __init__(self, dataset, *, drop_post_success: bool = True):
        self._dataset = dataset
        if not drop_post_success:
            self._index_map = list(range(len(dataset)))
            return
        keep = []
        # Iterate once at construction, reading only `cluster_label` to avoid
        # decoding videos.
        meta_table = dataset.hf_dataset.with_format(
            "numpy", columns=["cluster_label"]
        )
        for i, row in enumerate(meta_table):
            label = int(np.asarray(row["cluster_label"]).item())
            if label != -2:
                keep.append(i)
        logger.info(
            "Filter cluster_label != -2: kept %d / %d rows", len(keep), len(dataset)
        )
        self._index_map = keep

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, index: SupportsIndex) -> dict:
        return self._dataset[self._index_map[int(index)]]


# ----- Norm stats loader ------------------------------------------------------

def load_user_norm_stats(norm_stats_path: pathlib.Path) -> dict[str, _normalize.NormStats]:
    """Load the user-supplied norm_stats JSON.

    Accepts both the openpi standard format (top-level `{"norm_stats": {...}}`)
    and a flat `{"actions": ..., "state": ...}` form. Returns a `dict[str,
    NormStats]` directly usable as `data_config.norm_stats`.
    """
    raw_text = pathlib.Path(norm_stats_path).read_text()
    try:
        # Standard openpi format with a top-level "norm_stats" key.
        return _normalize.deserialize_json(raw_text)
    except Exception:
        pass
    # Fallback: tolerant manual parse (mean/std/optional q01/q99 per key).
    raw = json.loads(raw_text)
    inner = raw.get("norm_stats", raw)
    out: dict[str, _normalize.NormStats] = {}
    for k, v in inner.items():
        if not isinstance(v, dict) or "mean" not in v or "std" not in v:
            continue
        kwargs = dict(
            mean=np.asarray(v["mean"], dtype=np.float32),
            std=np.asarray(v["std"], dtype=np.float32),
        )
        if "q01" in v and v["q01"] is not None:
            kwargs["q01"] = np.asarray(v["q01"], dtype=np.float32)
        if "q99" in v and v["q99"] is not None:
            kwargs["q99"] = np.asarray(v["q99"], dtype=np.float32)
        out[k] = _normalize.NormStats(**kwargs)
    if not out:
        raise ValueError(f"Could not parse any norm stats from {norm_stats_path}")
    return out


# ----- Public factory ---------------------------------------------------------

def _resolve_repo_id(repo_id):
    if isinstance(repo_id, (list, tuple)):
        if len(repo_id) != 1:
            raise ValueError(
                f"Chunked dataloader expects exactly one task; got {len(repo_id)}: {repo_id!r}"
            )
        repo_id = repo_id[0]
    if repo_id is None or repo_id == "fake":
        raise ValueError(f"Chunked dataloader requires a real repo_id, got {repo_id!r}")
    return repo_id


class ChunkedDataLoader:
    """Yields `(Observation, actions, rl_dict)` tuples — same shape as the
    legacy `DataLoaderImpl` so train_redflow.py can do
    `obs, actions, rl = batch`.

    Scalar RL fields (cluster_advantage / cluster_label / pull_alpha /
    is_success / progress_value) are reshaped to `(B, 1)` so they broadcast
    cleanly against the per-step `fm_err: (B, ah)` in the loss formulas.
    """
    _RL_KEYS = (
        "cluster_advantage", "cluster_label", "pull_target_action",
        "pull_alpha", "is_success", "progress_value",
    )
    # These are stored per-row as shape [1] and arrive as (B,) after collate;
    # the loss formulas expect (B, 1).
    _SCALAR_RL_KEYS = (
        "cluster_advantage", "cluster_label", "pull_alpha", "is_success", "progress_value",
    )

    def __init__(self, data_config, torch_loader, dataset_size):
        self._data_config = data_config
        self._loader = torch_loader
        self._size = dataset_size

    def data_config(self):
        return self._data_config

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        for batch in self._loader:
            rl = {}
            for k in self._RL_KEYS:
                if k not in batch:
                    continue
                v = batch[k]
                if k in self._SCALAR_RL_KEYS and v.ndim == 1:
                    v = v[..., None]   # (B,) -> (B, 1)
                rl[k] = v
            yield _model.Observation.from_dict(batch), batch["actions"], rl


def create_chunked_data_loader(
    config: _config.TrainConfig,
    *,
    norm_stats_path: pathlib.Path,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = True,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
):
    """Top-level entrypoint for RedFlow chunked offline-RL training.

    Builds the standard `transform_dataset` pipeline but injects two custom
    transforms (`ReshapeChunkedActions`, `NormalizeAuxActions`)
    at the head of `data_transforms.inputs`, so they run after `repack_transforms`
    and before `AlohaInputs` (the pre-existing data_transform). The user's
    norm_stats JSON replaces `data_config.norm_stats` so `Normalize` uses the
    correct stats for the standard `actions` / `state` fields.
    """
    from openpi.training import data_loader as _stdloader

    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = _resolve_repo_id(data_config.repo_id)

    # 1. Load user-supplied norm_stats and override data_config.norm_stats.
    user_norm_stats = load_user_norm_stats(norm_stats_path)
    if "actions" not in user_norm_stats:
        raise KeyError(
            f"norm_stats at {norm_stats_path} is missing the 'actions' entry; "
            "needed by ReshapeChunkedActions / NormalizeAuxActions / Normalize."
        )

    # 2. Build custom transforms (run after repack, before AlohaInputs).
    #    Ordering: ReshapeChunkedActions -> NormalizeAuxActions
    action_stats = user_norm_stats["actions"]
    custom_inputs = [
        ReshapeChunkedActions(),
        NormalizeAuxActions(
            action_mean=np.asarray(action_stats.mean, dtype=np.float32),
            action_std=np.asarray(action_stats.std, dtype=np.float32),
        ),
    ]
    # AlohaInputs (and most subsequent transforms) build a fresh dict, dropping
    # the RL passthrough fields. Wrap each downstream transform with WithExtras
    # so RL fields are saved on entry and re-attached on exit.
    wrapped_data_transforms = tuple(WithExtras(t) for t in data_config.data_transforms.inputs)
    new_data_transforms = _transforms.Group(
        inputs=tuple(custom_inputs) + wrapped_data_transforms,
        outputs=data_config.data_transforms.outputs,
    )

    # Same WithExtras wrapping for model_transforms (tokenize prompt, resize
    # images — also build fresh dicts).
    wrapped_model_transforms = _transforms.Group(
        inputs=tuple(WithExtras(t) for t in data_config.model_transforms.inputs),
        outputs=data_config.model_transforms.outputs,
    )

    data_config = dataclasses.replace(
        data_config,
        norm_stats=user_norm_stats,
        data_transforms=new_data_transforms,
        model_transforms=wrapped_model_transforms,
    )

    # 3. Build the base dataset and apply prompt_from_task PRE-pipeline so
    #    repack_transforms can include `"prompt": "prompt"`.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    raw = lerobot_dataset.LeRobotDataset(repo_id)
    base = FilteredLeRobotDataset(raw, drop_post_success=True)
    if data_config.prompt_from_task:
        base = _stdloader.TransformedDataset(
            base, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    dataset_size = len(base)

    # 4. Build the full transform pipeline manually so we can wrap Normalize
    #    with WithExtras too (transform_dataset constructs Normalize internally
    #    and we can't otherwise intercept it). Order matches transform_dataset:
    #      repack → data_transforms.inputs → Normalize → model_transforms.inputs
    norm_stats = {} if skip_norm_stats else (user_norm_stats or {})
    full_inputs = (
        list(data_config.repack_transforms.inputs)
        + list(data_config.data_transforms.inputs)
        + [WithExtras(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))]
        + list(data_config.model_transforms.inputs)
    )
    dataset = _stdloader.TransformedDataset(base, full_inputs)

    # 5. Wrap with TorchDataLoader (batching + sharding + workers).
    torch_loader = _stdloader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
    )
    return ChunkedDataLoader(data_config, torch_loader, dataset_size)
