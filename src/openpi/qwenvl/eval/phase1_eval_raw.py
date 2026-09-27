from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any
from typing import Optional
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm
from lerobot.datasets.video_utils import encode_video_frames

from openpi.qwenvl.data._lerobot_dataset import attach_frame_metadata
from openpi.qwenvl.data._lerobot_dataset import iter_episode_frame_triplet_chunks
from openpi.qwenvl.data._lerobot_dataset import load_lerobot_dataset
from openpi.qwenvl.data._lerobot_dataset import load_lerobot_metadata
from openpi.qwenvl.data._lerobot_dataset import read_frame_rows
from openpi.qwenvl.data._lerobot_dataset import resolve_episode_local_row_range
from openpi.qwenvl.data._lerobot_dataset import resolve_parquet_path
from openpi.qwenvl.data._lerobot_dataset import resolve_prompt
from openpi.qwenvl.data.data_processor import parse_camera_names
from openpi.qwenvl.eval._artifacts import write_csv
from openpi.qwenvl.eval._artifacts import write_json
from openpi.qwenvl.eval._artifacts import write_jsonl
from openpi.qwenvl.modeling.pi06_inference import assert_strict_checkpoint_dir
from openpi.qwenvl.modeling.pi06_inference import load_pi06_inference_components
from openpi.qwenvl.modeling.pi06_inference import predict_pi06_values


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: evaluate raw value curves on a LeRobot dataset.")
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--model_type", default="pi06", choices=("pi06", "pair"))
    parser.add_argument("--camera_names", default="cam_high", type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--frame_interval", default=1, type=int)
    parser.add_argument("--relative_interval", default=50, type=int)
    parser.add_argument("--max_episodes", default=None, type=int)
    parser.add_argument("--episode_indices", nargs="*", default=None, type=int)
    parser.add_argument("--stream_chunk_size", default=128, type=int)
    parser.add_argument("--attn_implementation", default="flash_attention_2", type=str)
    parser.add_argument("--bf16", default=True, type=_str2bool)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--save_plots", default=True, type=_str2bool)
    parser.add_argument("--save_videos", default=False, type=_str2bool)
    parser.add_argument("--video_fps", default=30, type=int)
    return parser.parse_args(argv)


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _episode_stem(episode_idx: int) -> str:
    return f"episode_{int(episode_idx):06d}"


def _save_plot(path: Path, rows: list[dict[str, Any]], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [row["dataset_frame_index"] for row in rows]
    plt.figure(figsize=(12, 4.5))
    plt.plot(x_values, [row["absolute_value"] for row in rows], label="absolute_value", linewidth=1.6)
    plt.plot(x_values, [row["relative_advantage"] for row in rows], label="relative_advantage", linewidth=1.2)
    plt.plot(x_values, [row["absolute_advantage"] for row in rows], label="absolute_advantage", linewidth=1.2)
    plt.xlabel("frame_index")
    plt.ylabel("score")
    plt.ylim(-1.05, 1.05)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_phase1_episode_video(
    *,
    output_root: Path,
    task_name: str,
    episode_idx: int,
    rows: list[dict[str, Any]],
    metadata,
    dataset,
    camera_names: Sequence[str],
    frame_interval: int,
    stream_chunk_size: int,
    video_fps: int,
) -> Path:
    """Save a per-episode video with three camera views on top and value curve at bottom."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not camera_names:
        raise ValueError("camera_names must contain at least one camera.")
    display_camera_names = list(camera_names[:1])

    # Re-sample episode triplets for the specified cameras
    triplet_chunks = iter_episode_frame_triplet_chunks(
        dataset,
        metadata,
        int(episode_idx),
        display_camera_names,
        frame_interval=int(frame_interval),
        chunk_size=int(stream_chunk_size),
    )
    frame_views: list[tuple[Image.Image, ...]] = [
        frame_view for chunk in triplet_chunks for frame_view in chunk
    ]

    if not frame_views:
        raise ValueError(f"No frames found for episode {episode_idx} when saving video.")

    values = np.asarray([row["absolute_value"] for row in rows], dtype=np.float32)
    num_frames = min(len(frame_views), len(values))
    if num_frames <= 0:
        raise ValueError(f"Episode {episode_idx} has no frames for video generation.")

    frame_views = frame_views[:num_frames]
    values = values[:num_frames]

    episode_stem = _episode_stem(int(episode_idx))
    videos_root = output_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    frame_dir = videos_root / f"{episode_stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Prepare figure layout
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])
    ax_main = fig.add_subplot(gs[0, 0])
    ax_value = fig.add_subplot(gs[1, :])

    main0 = np.asarray(frame_views[0][0], dtype=np.uint8)
    im_main = ax_main.imshow(main0)
    ax_main.set_title(display_camera_names[0])
    ax_main.axis("off")

    x_values = np.arange(num_frames, dtype=np.int32)
    ax_value.plot(x_values, values, label="absolute_value", linewidth=1.5)
    cursor_line = ax_value.axvline(0, color="red", linewidth=1.0, label="current_frame")
    ax_value.set_xlabel("frame_index")
    ax_value.set_ylabel("value")
    ax_value.set_ylim(-1.05, 1.05)
    ax_value.set_title(f"{task_name} raw values episode {episode_idx}")
    ax_value.grid(True, alpha=0.3)
    ax_value.legend()

    fig.tight_layout()

    # Render frames
    for frame_idx, views in enumerate(frame_views):
        im_main.set_data(np.asarray(views[0], dtype=np.uint8))
        cursor_line.set_xdata([frame_idx, frame_idx])

        fig.canvas.draw()
        frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame_rgba[:, :, :3]  # Drop alpha channel
        Image.fromarray(frame, mode="RGB").save(frame_dir / f"frame-{frame_idx:06d}.png")

    plt.close(fig)

    video_path = videos_root / f"{episode_stem}.mp4"
    encode_video_frames(frame_dir, video_path, video_fps, vcodec="h264", overwrite=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    return video_path


def resolve_episode_indices(metadata, max_episodes: int | None, episode_indices: Sequence[int] | None) -> list[int]:
    total_episodes = len(metadata.episodes)
    if episode_indices:
        resolved = [int(index) for index in episode_indices]
        for episode_idx in resolved:
            if episode_idx < 0 or episode_idx >= total_episodes:
                raise IndexError(f"episode_idx out of range: {episode_idx} not in [0, {total_episodes})")
        return resolved

    limit = total_episodes if max_episodes is None else min(total_episodes, int(max_episodes))
    return list(range(limit))


def _build_metric_rows(absolute_values: np.ndarray, relative_interval: int, episode_idx: int) -> list[dict[str, Any]]:
    relative_interval = max(1, int(relative_interval))
    num_frames = int(len(absolute_values))
    future_indices = np.minimum(np.arange(num_frames) + relative_interval, max(0, num_frames - 1))
    rows = []
    for frame_idx in range(num_frames):
        future_idx = int(future_indices[frame_idx])
        gap = future_idx - frame_idx
        if gap <= 0:
            derived_adv = 0.0
        else:
            derived_adv = float(absolute_values[future_idx] - absolute_values[frame_idx])
            if gap != relative_interval:
                derived_adv = derived_adv / gap * relative_interval
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "future_frame_idx": int(future_idx),
                "absolute_value": _clip_unit(float(absolute_values[frame_idx])),
                "relative_advantage": _clip_unit(derived_adv),
                "absolute_advantage": _clip_unit(derived_adv),
                "model_type": "pi06",
                "episode_idx": int(episode_idx),
            }
        )
    return rows


def evaluate_phase1_episode(
    *,
    metadata,
    dataset,
    components: dict[str, Any],
    repo_root: str | Path,
    episode_idx: int,
    model_type: str,
    prompt_override: str | None,
    camera_names: Sequence[str],
    batch_size: int,
    frame_interval: int,
    relative_interval: int,
    stream_chunk_size: int,
) -> tuple[list[dict[str, Any]], str, Path]:
    if model_type != "pi06":
        raise NotImplementedError("Phase 1 eval currently supports pi06 only. Pair eval is not part of this validation pass.")

    prompt = resolve_prompt(metadata, int(episode_idx), prompt_override)
    src_parquet = resolve_parquet_path(repo_root, metadata, int(episode_idx))
    row_start, row_end = resolve_episode_local_row_range(metadata, int(episode_idx))
    frame_rows = read_frame_rows(src_parquet, row_start=row_start, row_end=row_end)
    sampled_frame_rows = frame_rows[:: max(1, int(frame_interval))]

    # Compute the total number of chunks for the progress bar.
    total_frames = len(sampled_frame_rows)
    chunk_size = max(1, int(stream_chunk_size))
    total_chunks = (total_frames + chunk_size - 1) // chunk_size  # Round up

    absolute_chunks: list[np.ndarray] = []
    consumed_rows = 0
    chunk_iter = iter_episode_frame_triplet_chunks(
        dataset,
        metadata,
        int(episode_idx),
        camera_names,
        frame_interval=int(frame_interval),
        chunk_size=int(stream_chunk_size),
    )
    with tqdm(
        chunk_iter,
        total=total_chunks,
        desc=f"Episode {episode_idx}",
        unit="chunk",
        leave=False,
    ) as pbar:
        for frame_triplet_chunk in pbar:
            absolute_chunk = predict_pi06_values(
                components,
                prompt,
                frame_triplet_chunk,
                batch_size=int(batch_size),
            ).astype(np.float32)
            chunk_frame_rows = sampled_frame_rows[consumed_rows : consumed_rows + len(absolute_chunk)]
            if len(chunk_frame_rows) != len(absolute_chunk):
                raise ValueError(
                    f"Episode {episode_idx} streaming mismatch: predictions={len(absolute_chunk)} "
                    f"frame_rows={len(chunk_frame_rows)}"
                )
            absolute_chunks.append(absolute_chunk)
            consumed_rows += len(absolute_chunk)
            # Update the progress bar with the number of processed frames.
            pbar.set_postfix({"frames": f"{consumed_rows}/{total_frames}"})
    if consumed_rows != len(sampled_frame_rows):
        raise ValueError(
            f"Episode {episode_idx} did not consume all sampled rows: consumed={consumed_rows} "
            f"expected={len(sampled_frame_rows)}"
        )

    absolute_values = np.concatenate(absolute_chunks, axis=0) if absolute_chunks else np.zeros((0,), dtype=np.float32)
    rows = _build_metric_rows(absolute_values, int(relative_interval), int(episode_idx))
    rows = attach_frame_metadata(rows, sampled_frame_rows)
    return rows, prompt, src_parquet


def write_phase1_episode(
    *,
    output_root: Path,
    task_name: str,
    episode_idx: int,
    rows: list[dict[str, Any]],
    prompt: str,
    src_parquet: Path,
    save_plots: bool,
) -> dict[str, Any]:
    episode_stem = _episode_stem(int(episode_idx))
    metrics_root = output_root / "metrics"
    plots_root = output_root / "plots"
    metrics_jsonl_path = metrics_root / f"{episode_stem}.jsonl"
    metrics_csv_path = metrics_root / f"{episode_stem}.csv"
    plot_path = plots_root / f"{episode_stem}.png"
    write_jsonl(metrics_jsonl_path, rows)
    write_csv(metrics_csv_path, rows)
    if save_plots:
        _save_plot(plot_path, rows, title=f"{task_name} raw values episode {episode_idx}")
    return {
        "episode_idx": int(episode_idx),
        "prompt": prompt,
        "num_frames": int(len(rows)),
        "metrics_jsonl": str(metrics_jsonl_path.relative_to(output_root)),
        "metrics_csv": str(metrics_csv_path.relative_to(output_root)),
        "plot": str(plot_path.relative_to(output_root)) if save_plots else None,
        "video": None,
        "source_parquet": str(src_parquet),
    }


def write_phase1_manifest(
    *,
    output_root: Path,
    task_name: str,
    model_type: str,
    repo_root: str | Path,
    checkpoint_dir: str | Path,
    camera_names: Sequence[str],
    batch_size: int,
    frame_interval: int,
    relative_interval: int,
    stream_chunk_size: int,
    episode_records: list[dict[str, Any]],
):
    write_json(
        output_root / "manifest.json",
        {
            "phase": 1,
            "task_name": task_name,
            "model_type": str(model_type),
            "repo_root": str(Path(repo_root).expanduser().resolve()),
            "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
            "params": {
                "camera_names": list(camera_names),
                "batch_size": int(batch_size),
                "frame_interval": int(frame_interval),
                "relative_interval": int(relative_interval),
                "stream_chunk_size": int(stream_chunk_size),
            },
            "episodes": episode_records,
        },
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.model_type != "pi06":
        raise NotImplementedError("Phase 1 eval currently supports pi06 only. Pair eval is not part of this validation pass.")
    checkpoint_dir = assert_strict_checkpoint_dir(args.checkpoint_dir)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    metadata = load_lerobot_metadata(repo_root)
    dataset = load_lerobot_dataset(repo_root)
    camera_names = parse_camera_names(args.camera_names)
    stream_chunk_size = max(int(args.stream_chunk_size), int(args.batch_size), 1)
    episode_indices = resolve_episode_indices(metadata, args.max_episodes, args.episode_indices)

    components = load_pi06_inference_components(
        checkpoint_dir,
        attn_implementation=args.attn_implementation,
        bf16=args.bf16,
        device=args.device,
    )

    manifest_episodes: list[dict[str, Any]] = []
    for episode_idx in tqdm(episode_indices, desc="Phase1 eval raw"):
        rows, prompt, src_parquet = evaluate_phase1_episode(
            metadata=metadata,
            dataset=dataset,
            components=components,
            repo_root=repo_root,
            episode_idx=int(episode_idx),
            model_type=args.model_type,
            prompt_override=args.prompt,
            camera_names=camera_names,
            batch_size=int(args.batch_size),
            frame_interval=int(args.frame_interval),
            relative_interval=int(args.relative_interval),
            stream_chunk_size=stream_chunk_size,
        )
        record = write_phase1_episode(
            output_root=output_root,
            task_name=args.task_name,
            episode_idx=int(episode_idx),
            rows=rows,
            prompt=prompt,
            src_parquet=src_parquet,
            save_plots=bool(args.save_plots),
        )
        if args.save_videos:
            video_path = save_phase1_episode_video(
                output_root=output_root,
                task_name=args.task_name,
                episode_idx=int(episode_idx),
                rows=rows,
                metadata=metadata,
                dataset=dataset,
                camera_names=camera_names,
                frame_interval=int(args.frame_interval),
                stream_chunk_size=stream_chunk_size,
                video_fps=int(args.video_fps),
            )
            record["video"] = str(video_path.relative_to(output_root))
        manifest_episodes.append(record)

    write_phase1_manifest(
        output_root=output_root,
        task_name=args.task_name,
        model_type=args.model_type,
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        camera_names=camera_names,
        batch_size=int(args.batch_size),
        frame_interval=int(args.frame_interval),
        relative_interval=int(args.relative_interval),
        stream_chunk_size=stream_chunk_size,
        episode_records=manifest_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
