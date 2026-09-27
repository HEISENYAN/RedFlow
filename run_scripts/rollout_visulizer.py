#!/usr/bin/env python3
"""
Convert rollout HDF5 episodes to videos using head camera frames.

Usage:
  python3 rollout_visulizer.py /path/to/output_root
"""

import argparse
from pathlib import Path
from typing import List, Optional

import h5py
import imageio.v2 as imageio
import numpy as np


CAMERA_DATASET_CANDIDATES = [
    "/observation/images/head_camera",
    "/observation/head_camera/rgb",
    "/observation/images/cam_high",  # Fallback for current rollout schema.
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search task folders and convert episode_*.hdf5 to mp4 using head camera frames."
    )
    parser.add_argument("root", type=str, help="Root directory containing task folders.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS (default: 20).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos if they already exist.",
    )
    return parser.parse_args()


def find_episode_files(root: Path) -> List[Path]:
    return sorted(root.rglob("episode_*.hdf5"))


def find_camera_dataset(file_handle: h5py.File) -> Optional[h5py.Dataset]:
    for dataset_path in CAMERA_DATASET_CANDIDATES:
        if dataset_path in file_handle:
            dataset = file_handle[dataset_path]
            if isinstance(dataset, h5py.Dataset):
                return dataset

    # Generic fallback if dataset path naming differs.
    fallback: List[h5py.Dataset] = []

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        lname = name.lower()
        if "head_camera" in lname or "cam_high" in lname:
            fallback.append(obj)

    file_handle.visititems(visit)
    if fallback:
        fallback.sort(key=lambda ds: ds.name)
        return fallback[0]
    return None


def decode_frame(frame: np.ndarray) -> np.ndarray:
    frame_np = np.asarray(frame)

    # CHW -> HWC
    if frame_np.ndim == 3 and frame_np.shape[0] in (1, 3) and frame_np.shape[2] not in (1, 3):
        frame_np = np.transpose(frame_np, (1, 2, 0))

    # Single channel -> 3 channel
    if frame_np.ndim == 2:
        frame_np = np.repeat(frame_np[..., None], 3, axis=2)
    elif frame_np.ndim == 3 and frame_np.shape[2] == 1:
        frame_np = np.repeat(frame_np, 3, axis=2)

    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    return frame_np


def export_episode_video(hdf5_path: Path, fps: int, overwrite: bool) -> None:
    out_path = hdf5_path.with_suffix(".mp4")
    if out_path.exists() and not overwrite:
        print(f"[SKIP] {hdf5_path} -> {out_path} (already exists)")
        return

    with h5py.File(hdf5_path, "r") as f:
        dataset = find_camera_dataset(f)
        if dataset is None:
            print(f"[WARN] {hdf5_path} no head_camera/cam_high dataset found")
            return
        if len(dataset.shape) == 0 or dataset.shape[0] == 0:
            print(f"[WARN] {hdf5_path} empty camera dataset: {dataset.name}")
            return

        print(f"[INFO] {hdf5_path} using {dataset.name}, frames={dataset.shape[0]}")
        with imageio.get_writer(str(out_path), fps=fps, codec="libx264") as writer:
            for idx in range(dataset.shape[0]):
                writer.append_data(decode_frame(dataset[idx]))
    print(f"[OK]   {out_path}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}")
        return 1
    if not root.is_dir():
        print(f"[ERROR] Root path is not a directory: {root}")
        return 1

    episodes = find_episode_files(root)
    if not episodes:
        print(f"[WARN] No episode_*.hdf5 found under: {root}")
        return 0

    print(f"[INFO] Found {len(episodes)} episode files under {root}")
    for hdf5_path in episodes:
        export_episode_video(hdf5_path, fps=args.fps, overwrite=args.overwrite)
    print("[DONE] All available episodes processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
