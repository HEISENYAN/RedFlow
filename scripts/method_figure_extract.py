"""Extract one success + one fail episode from a real_rollout task and save:

  - Every original-fps frame of all 3 cameras as PNGs
  - Chunk-level progress curve as a matplotlib figure

Usage:
  python scripts/method_figure_extract.py \\
      --task folding_clothes \\
      --out ./method_figure \\
      --seed 0
  python scripts/method_figure_extract.py --task auto       # randomly pick a task
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = pathlib.Path(os.environ.get("REAL_ROLLOUT_ROOT", "./datasets/real_rollout"))
ALL_TASKS = ("folding_clothes", "table_cleaning", "table_sweeping")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def load_chunked_table(task_dir: pathlib.Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(task_dir / "lerobot_chunked" / "data" / "chunk-000" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def pick_episodes(df: pd.DataFrame, rng: random.Random) -> tuple[int, int]:
    """Pick one random success + one random fail episode_index from the parquet table."""
    ep_success = df.groupby("episode_index")["is_success"].first()
    success_eps = ep_success[ep_success].index.tolist()
    fail_eps = ep_success[~ep_success].index.tolist()
    if not success_eps:
        raise ValueError("No success episodes found")
    if not fail_eps:
        raise ValueError("No fail episodes found")
    return int(rng.choice(success_eps)), int(rng.choice(fail_eps))


def video_path(task_dir: pathlib.Path, camera: str, episode_idx: int) -> pathlib.Path:
    return (
        task_dir / "lerobot_chunked" / "videos" /
        f"observation.images.{camera}" / "chunk-000" / f"file-{episode_idx:03d}.mp4"
    )


def decode_video_frames(path: pathlib.Path) -> list[np.ndarray]:
    """Return all frames as a list of HxWx3 uint8 BGR arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cv2 cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_episode_frames(task_dir: pathlib.Path, episode_idx: int, out_dir: pathlib.Path) -> dict[str, int]:
    """For one episode, save every frame of every camera. Returns per-camera frame counts."""
    counts = {}
    for cam in CAMERAS:
        src = video_path(task_dir, cam, episode_idx)
        if not src.exists():
            logger.warning("Missing %s — skipping this camera", src)
            counts[cam] = 0
            continue
        cam_dir = out_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        frames = decode_video_frames(src)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(cam_dir / f"{i:06d}.png"), frame)
        counts[cam] = len(frames)
        logger.info("  %s: %d frames → %s", cam, len(frames), cam_dir)
    return counts


def progress_for_episode(df: pd.DataFrame, episode_idx: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (timestamps_sec, progress_values, is_success) for one episode at chunk level."""
    sub = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    # frame_index here is the chunk index (1 chunk = 50 original frames @50fps = 1 sec)
    t = sub["frame_index"].astype(float).to_numpy()
    p = sub["progress_value"].apply(lambda x: float(np.asarray(x).item())).to_numpy()
    is_success = bool(np.asarray(sub.iloc[0]["is_success"]).item())
    return t, p, is_success


def plot_progress(success_data, fail_data, out_path: pathlib.Path):
    """Plot success + fail progress curves on one axis."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)

    t_s, p_s, ep_s = success_data
    ax.plot(t_s, p_s, "-o", color="#2a9d8f", linewidth=2, markersize=4,
            label=f"success (ep {ep_s})")
    t_f, p_f, ep_f = fail_data
    ax.plot(t_f, p_f, "-o", color="#e76f51", linewidth=2, markersize=4,
            label=f"fail (ep {ep_f})")

    ax.set_xlabel("chunk index (≈ seconds, 1 chunk = 50 frames @ 50 Hz)")
    ax.set_ylabel("progress_value")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", frameon=False)
    ax.set_title("GRM progress over time")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Saved progress plot → %s (and .pdf)", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="auto",
                    help="One of folding_clothes / table_cleaning / table_sweeping, or 'auto' to pick randomly")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("./method_figure"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--success_ep", type=int, default=None,
                    help="Pin a specific success episode_index instead of random")
    ap.add_argument("--fail_ep", type=int, default=None,
                    help="Pin a specific fail episode_index instead of random")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.task == "auto":
        task = rng.choice(ALL_TASKS)
        logger.info("Picked task=%s (seed=%d)", task, args.seed)
    else:
        task = args.task
        if task not in ALL_TASKS:
            raise ValueError(f"task must be one of {ALL_TASKS}, got {task}")

    task_dir = DATASET_ROOT / task
    df = load_chunked_table(task_dir)
    logger.info("%s: %d rows, %d episodes", task, len(df), df["episode_index"].nunique())

    if args.success_ep is None or args.fail_ep is None:
        s_pick, f_pick = pick_episodes(df, rng)
        success_ep = args.success_ep if args.success_ep is not None else s_pick
        fail_ep = args.fail_ep if args.fail_ep is not None else f_pick
    else:
        success_ep, fail_ep = args.success_ep, args.fail_ep
    logger.info("Picked success_ep=%d, fail_ep=%d", success_ep, fail_ep)

    out_root = args.out / task
    out_root.mkdir(parents=True, exist_ok=True)

    # === Save per-frame PNGs ===
    for label, ep in (("success", success_ep), ("fail", fail_ep)):
        out_dir = out_root / f"{label}_ep{ep:03d}"
        logger.info("=== %s episode %d → %s ===", label, ep, out_dir)
        save_episode_frames(task_dir, ep, out_dir)

    # === Plot progress curves ===
    success_data = progress_for_episode(df, success_ep) + (success_ep,)
    fail_data = progress_for_episode(df, fail_ep) + (fail_ep,)
    # progress_for_episode returns (t, p, is_success); we want last element to be ep_idx
    # rebuild correctly:
    ts, ps, _ = progress_for_episode(df, success_ep)
    tf, pf, _ = progress_for_episode(df, fail_ep)
    plot_progress((ts, ps, success_ep), (tf, pf, fail_ep), out_root / "progress_curve.png")

    # === Summary ===
    logger.info("==========")
    logger.info("Done. task=%s   success_ep=%d   fail_ep=%d", task, success_ep, fail_ep)
    logger.info("PNG frames + progress_curve.png/.pdf under %s", out_root)


if __name__ == "__main__":
    main()
