"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# debugpy.wait_for_client()
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch
import tensorflow_datasets as tfds

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        # "left_waist",
        # "left_shoulder",
        # "left_elbow",
        # "left_forearm_roll",
        # "left_wrist_angle",
        # "left_wrist_rotate",
        # "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        #"cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 256, 256), # TODO modified here, was originally 3, 480, 640
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118




def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # decode as color image
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:,7:14])
        action = torch.from_numpy(ep["/action"][:,7:14])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:,7:14])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:,7:14])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                #"cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    tfds_ds = tfds.load("coffee:1.0.0", split="train", data_dir="./processed_data")
    episodes = tfds_ds.take(300)

    for episode in episodes:
        for step in episode["steps"].as_numpy_iterator():
            frame = {
                "observation.state": step["observation"]["state"],
                "action": step["action"],
                "task": step["language_instruction"].decode(),
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_aloha)
