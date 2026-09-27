from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from openpi.qwenvl.eval._artifacts import read_json
from openpi.qwenvl.eval._artifacts import read_jsonl
from openpi.qwenvl.eval._artifacts import write_json
from openpi.qwenvl.eval.phase1_eval_raw import evaluate_phase1_episode
from openpi.qwenvl.eval.phase1_eval_raw import resolve_episode_indices
from openpi.qwenvl.eval.phase1_eval_raw import save_phase1_episode_video
from openpi.qwenvl.eval.phase1_eval_raw import write_phase1_episode
from openpi.qwenvl.eval.phase1_eval_raw import write_phase1_manifest
from openpi.qwenvl.eval.phase1_eval_raw import main as phase1_main
from openpi.qwenvl.eval.phase2_smooth import smooth_phase2_episode_rows
from openpi.qwenvl.eval.phase2_smooth import write_phase2_episode
from openpi.qwenvl.eval.phase2_smooth import write_phase2_manifest
from openpi.qwenvl.eval.phase2_smooth import main as phase2_main
from openpi.qwenvl.eval.phase3_segment import segment_phase3_episode_rows
from openpi.qwenvl.eval.phase3_segment import write_phase3_episode
from openpi.qwenvl.eval.phase3_segment import write_phase3_manifest
from openpi.qwenvl.eval.phase3_segment import main as phase3_main
from openpi.qwenvl.eval.phase4_rewrite import _copy_untouched_parquet_files
from openpi.qwenvl.eval.phase4_rewrite import init_phase4_context
from openpi.qwenvl.eval.phase4_rewrite import rewrite_phase4_episode
from openpi.qwenvl.eval.phase4_rewrite import _update_meta_info
from openpi.qwenvl.eval.phase4_rewrite import _update_meta_stats
from openpi.qwenvl.eval.phase4_rewrite import write_phase4_manifest
from openpi.qwenvl.eval.phase4_rewrite import main as phase4_main
from openpi.qwenvl.modeling.pi06_inference import assert_strict_checkpoint_dir
from openpi.qwenvl.modeling.pi06_inference import load_pi06_inference_components
from openpi.qwenvl.modeling.pair_inference import load_pair_inference_components
from openpi.qwenvl.data._lerobot_dataset import load_lerobot_dataset
from openpi.qwenvl.data._lerobot_dataset import load_lerobot_metadata
from openpi.qwenvl.data.data_processor import parse_camera_names
from openpi.qwenvl.utils.convert_hdf5_to_lerobot import convert_task_hdf5_to_lerobot


def _log_progress(message: str):
    print(f"[pipeline] {message}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict phase0-4 QwenVL eval pipeline for a single task.")
    parser.add_argument("--pipeline_mode", default="episode", choices=("episode", "stage"))
    parser.add_argument("--model_type", default="pi06", choices=("pi06", "pair"))
    parser.add_argument("--source_mode", required=True, choices=("hdf5", "lerobot"))
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--start_phase", default=0, type=int)
    parser.add_argument("--end_phase", default=4, type=int)
    parser.add_argument("--camera_names", default="cam_high", type=str)
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--mode", default="video", choices=("video", "image"))
    parser.add_argument("--fps", default=20, type=int)
    parser.add_argument("--robot_type", default="aloha", type=str)
    parser.add_argument("--max_episodes_per_task", default=None, type=int)
    parser.add_argument("--max_episodes", default=None, type=int)
    parser.add_argument("--episode_indices", nargs="*", default=None, type=int)
    parser.add_argument("--num_workers_per_task", default=1, type=int, help="Number of worker processes for parallel episode processing in phase0 conversion")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--stream_chunk_size", default=128, type=int)
    parser.add_argument("--frame_interval", default=1, type=int)
    parser.add_argument("--relative_interval", default=50, type=int)
    parser.add_argument("--save_plots", default=True, type=lambda value: str(value).lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--save_videos", default=False, type=lambda value: str(value).lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--video_fps", default=30, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--attn_implementation", default="flash_attention_2", type=str)
    parser.add_argument("--bf16", default=True, type=lambda value: str(value).lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--savgol_window", default=11, type=int)
    parser.add_argument("--savgol_polyorder", default=2, type=int)
    parser.add_argument("--peak_prominence", default=0.08, type=float)
    parser.add_argument("--peak_distance_ratio", default=0.05, type=float)
    parser.add_argument("--min_rise_amplitude", default=0.10, type=float)
    parser.add_argument("--success_threshold", default=0.98, type=float)
    parser.add_argument("--value_column", default="value", type=str)
    parser.add_argument("--final_dir", default=None, type=str, help="Final output directory. If provided, dataset will be copied to final_dir/task_name")
    parser.add_argument("--resume", default=False, type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
                        help="If true, reuse existing per-episode artifacts under output_root to skip finished episodes")
    return parser.parse_args(argv)


def _run_stage_pipeline(args: argparse.Namespace, current_repo_root: Path, output_root: Path):
    if args.start_phase <= 1 <= args.end_phase:
        phase_start = time.perf_counter()
        _log_progress(f"phase1 start task={args.task_name}")
        phase1_main(
            [
                "--checkpoint_dir",
                args.checkpoint_dir,
                "--model_type",
                args.model_type,
                "--repo_root",
                str(current_repo_root),
                "--output_root",
                str(output_root / "phase1_eval_raw"),
                "--task_name",
                args.task_name,
                "--camera_names",
                args.camera_names,
                "--batch_size",
                str(args.batch_size),
                "--frame_interval",
                str(args.frame_interval),
                "--relative_interval",
                str(args.relative_interval),
                "--stream_chunk_size",
                str(args.stream_chunk_size),
                "--attn_implementation",
                args.attn_implementation,
                "--bf16",
                str(bool(args.bf16)).lower(),
                "--save_plots",
                str(bool(args.save_plots)).lower(),
                "--save_videos",
                str(bool(args.save_videos)).lower(),
                "--video_fps",
                str(args.video_fps),
                *([] if args.prompt is None else ["--prompt", args.prompt]),
                *([] if args.max_episodes is None else ["--max_episodes", str(args.max_episodes)]),
                *([] if not args.episode_indices else ["--episode_indices", *[str(value) for value in args.episode_indices]]),
                *([] if args.device is None else ["--device", args.device]),
            ]
        )
        _log_progress(f"phase1 done task={args.task_name} elapsed={time.perf_counter() - phase_start:.2f}s")
    if args.start_phase <= 2 <= args.end_phase:
        phase_start = time.perf_counter()
        _log_progress(f"phase2 start task={args.task_name}")
        phase2_main(
            [
                "--phase1_root",
                str(output_root / "phase1_eval_raw"),
                "--output_root",
                str(output_root / "phase2_smooth"),
                "--task_name",
                args.task_name,
                "--savgol_window",
                str(args.savgol_window),
                "--savgol_polyorder",
                str(args.savgol_polyorder),
                "--save_plots",
                str(bool(args.save_plots)).lower(),
            ]
        )
        _log_progress(f"phase2 done task={args.task_name} elapsed={time.perf_counter() - phase_start:.2f}s")
    if args.start_phase <= 3 <= args.end_phase:
        phase_start = time.perf_counter()
        _log_progress(f"phase3 start task={args.task_name}")
        phase3_main(
            [
                "--phase2_root",
                str(output_root / "phase2_smooth"),
                "--output_root",
                str(output_root / "phase3_segment"),
                "--task_name",
                args.task_name,
                "--savgol_window",
                str(args.savgol_window),
                "--savgol_polyorder",
                str(args.savgol_polyorder),
                "--peak_prominence",
                str(args.peak_prominence),
                "--peak_distance_ratio",
                str(args.peak_distance_ratio),
                "--min_rise_amplitude",
                str(args.min_rise_amplitude),
                "--success_threshold",
                str(args.success_threshold),
                "--save_plots",
                str(bool(args.save_plots)).lower(),
            ]
        )
        _log_progress(f"phase3 done task={args.task_name} elapsed={time.perf_counter() - phase_start:.2f}s")
    if args.start_phase <= 4 <= args.end_phase:
        phase_start = time.perf_counter()
        _log_progress(f"phase4 start task={args.task_name}")
        phase4_args = [
            "--phase3_root",
            str(output_root / "phase3_segment"),
            "--output_root",
            str(output_root / "phase4_rewrite"),
            "--task_name",
            args.task_name,
            "--value_column",
            args.value_column,
        ]
        if args.final_dir is not None:
            phase4_args.extend(["--final_dir", str(args.final_dir)])
        phase4_main(phase4_args)
        _log_progress(f"phase4 done task={args.task_name} elapsed={time.perf_counter() - phase_start:.2f}s")


def _run_episode_pipeline(args: argparse.Namespace, current_repo_root: Path, output_root: Path):
    if args.start_phase not in {0, 1}:
        raise ValueError(
            f"pipeline_mode=episode only supports start_phase 0 or 1, got start_phase={args.start_phase}"
        )
    checkpoint_dir = assert_strict_checkpoint_dir(args.checkpoint_dir)
    metadata = load_lerobot_metadata(current_repo_root)
    dataset = load_lerobot_dataset(current_repo_root)
    if args.model_type == "pi06":
        components = load_pi06_inference_components(
            checkpoint_dir,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            device=args.device,
        )
    else:
        components = load_pair_inference_components(
            checkpoint_dir,
            attn_implementation=args.attn_implementation,
            bf16=args.bf16,
            device=args.device,
        )
    camera_names = parse_camera_names(args.camera_names)
    stream_chunk_size = max(int(args.batch_size), int(args.stream_chunk_size), 1)
    episode_indices = resolve_episode_indices(metadata, args.max_episodes, args.episode_indices)

    phase1_records: list[dict[str, object]] = []
    phase2_records: list[dict[str, object]] = []
    phase3_records: list[dict[str, object]] = []
    phase4_records: list[dict[str, object]] = []
    rewritten_parquets: set[Path] = set()
    export_dataset_root = None
    phase4_metadata = None

    # === Preload existing phase manifests for resuming ===
    phase1_root = output_root / "phase1_eval_raw"
    phase2_root = output_root / "phase2_smooth"
    phase3_root = output_root / "phase3_segment"
    phase4_root = output_root / "phase4_rewrite"

    existing_phase1_by_idx: dict[int, dict[str, object]] = {}
    existing_phase2_by_idx: dict[int, dict[str, object]] = {}
    existing_phase3_by_idx: dict[int, dict[str, object]] = {}
    existing_phase4_by_idx: dict[int, dict[str, object]] = {}

    if args.resume:
        if (phase1_root / "manifest.json").exists():
            m1 = read_json(phase1_root / "manifest.json")
            for rec in m1.get("episodes", []):
                # Verify that the files actually exist.
                metrics_path = phase1_root / str(rec.get("metrics_jsonl", ""))
                if metrics_path.exists():
                    existing_phase1_by_idx[int(rec["episode_idx"])] = rec
        if (phase2_root / "manifest.json").exists():
            m2 = read_json(phase2_root / "manifest.json")
            for rec in m2.get("episodes", []):
                # Verify that the files actually exist.
                rows_jsonl = phase2_root / str(rec.get("rows_jsonl", ""))
                if rows_jsonl.exists():
                    existing_phase2_by_idx[int(rec["episode_idx"])] = rec
        if (phase3_root / "manifest.json").exists():
            m3 = read_json(phase3_root / "manifest.json")
            for rec in m3.get("episodes", []):
                # Verify that the files actually exist.
                rows_jsonl = phase3_root / str(rec.get("rows_jsonl", ""))
                if rows_jsonl.exists():
                    existing_phase3_by_idx[int(rec["episode_idx"])] = rec
        if (phase4_root / "manifest.json").exists():
            m4 = read_json(phase4_root / "manifest.json")
            for rec in m4.get("episodes", []):
                # Phase 4 records need no separate file check because results are written back to parquet.
                existing_phase4_by_idx[int(rec["episode_idx"])] = rec
                # Track rewritten parquet files for the later _copy_untouched_parquet_files call.
                rewritten_parquets.add(Path(str(rec["source_parquet"])))

    if args.end_phase >= 4:
        export_dataset_root, phase4_metadata = init_phase4_context(
            current_repo_root, output_root / "phase4_rewrite", task_name=args.task_name, final_dir=args.final_dir
        )

    for episode_idx in episode_indices:
        episode_start = time.perf_counter()
        _log_progress(f"episode start task={args.task_name} episode={episode_idx}")

        # ---------------- phase1: prediction ----------------
        if args.end_phase >= 1:
            # When resuming, load existing phase 1 results for this episode.
            if args.resume and int(episode_idx) in existing_phase1_by_idx:
                phase1_record = existing_phase1_by_idx[int(episode_idx)]
                metrics_path = phase1_root / str(phase1_record["metrics_jsonl"])
                # Recheck file existence in case files were deleted during the loop.
                if metrics_path.exists():
                    rows = read_jsonl(metrics_path)
                    prompt = str(phase1_record["prompt"])
                    src_parquet = Path(str(phase1_record["source_parquet"]))
                    if args.save_videos:
                        video_path = save_phase1_episode_video(
                            output_root=phase1_root,
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
                        phase1_record["video"] = str(video_path.relative_to(phase1_root))
                    phase1_records.append(phase1_record)
                    _log_progress(
                        f"phase1 episode skipped (resume) task={args.task_name} episode={episode_idx} "
                        f"frames={len(rows)}"
                    )
                else:
                    # Recompute if the files are missing.
                    _log_progress(
                        f"phase1 episode file missing, recomputing task={args.task_name} episode={episode_idx}"
                    )
                    rows, prompt, src_parquet = evaluate_phase1_episode(
                        metadata=metadata,
                        dataset=dataset,
                        components=components,
                        repo_root=current_repo_root,
                        episode_idx=int(episode_idx),
                        model_type=args.model_type,
                        prompt_override=args.prompt,
                        camera_names=camera_names,
                        batch_size=int(args.batch_size),
                        frame_interval=int(args.frame_interval),
                        relative_interval=int(args.relative_interval),
                        stream_chunk_size=stream_chunk_size,
                    )
                    phase1_record = write_phase1_episode(
                        output_root=phase1_root,
                        task_name=args.task_name,
                        episode_idx=int(episode_idx),
                        rows=rows,
                        prompt=prompt,
                        src_parquet=src_parquet,
                        save_plots=bool(args.save_plots),
                    )
                    if args.save_videos:
                        video_path = save_phase1_episode_video(
                            output_root=phase1_root,
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
                        phase1_record["video"] = str(video_path.relative_to(phase1_root))
                    phase1_records.append(phase1_record)
                    _log_progress(
                        f"phase1 episode done task={args.task_name} episode={episode_idx} "
                        f"frames={len(rows)} elapsed={time.perf_counter() - episode_start:.2f}s"
                    )
            else:
                # Original computation logic
                rows, prompt, src_parquet = evaluate_phase1_episode(
                    metadata=metadata,
                    dataset=dataset,
                    components=components,
                    repo_root=current_repo_root,
                    episode_idx=int(episode_idx),
                    model_type=args.model_type,
                    prompt_override=args.prompt,
                    camera_names=camera_names,
                    batch_size=int(args.batch_size),
                    frame_interval=int(args.frame_interval),
                    relative_interval=int(args.relative_interval),
                    stream_chunk_size=stream_chunk_size,
                )
                phase1_record = write_phase1_episode(
                    output_root=phase1_root,
                    task_name=args.task_name,
                    episode_idx=int(episode_idx),
                    rows=rows,
                    prompt=prompt,
                    src_parquet=src_parquet,
                    save_plots=bool(args.save_plots),
                )
                if args.save_videos:
                    video_path = save_phase1_episode_video(
                        output_root=phase1_root,
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
                    phase1_record["video"] = str(video_path.relative_to(phase1_root))
                phase1_records.append(phase1_record)
                _log_progress(
                    f"phase1 episode done task={args.task_name} episode={episode_idx} "
                    f"frames={len(rows)} elapsed={time.perf_counter() - episode_start:.2f}s"
                )

            # Flush the manifest after each episode, matching the original behavior.
            write_phase1_manifest(
                output_root=phase1_root,
                task_name=args.task_name,
                model_type=args.model_type,
                repo_root=current_repo_root,
                checkpoint_dir=checkpoint_dir,
                camera_names=camera_names,
                batch_size=int(args.batch_size),
                frame_interval=int(args.frame_interval),
                relative_interval=int(args.relative_interval),
                stream_chunk_size=stream_chunk_size,
                episode_records=phase1_records,
            )
        else:
            # If end_phase < 1, subsequent phases would not run anyway.
            continue

        if args.end_phase < 2:
            continue

        # ---------------- phase2: smoothing ----------------
        if args.resume and int(episode_idx) in existing_phase2_by_idx:
            phase2_record = existing_phase2_by_idx[int(episode_idx)]
            rows_jsonl = phase2_root / str(phase2_record["rows_jsonl"])
            # Recheck file existence.
            if rows_jsonl.exists():
                phase2_rows = read_jsonl(rows_jsonl)
                phase2_records.append(phase2_record)
                _log_progress(
                    f"phase2 episode skipped (resume) task={args.task_name} episode={episode_idx} "
                    f"elapsed={time.perf_counter() - episode_start:.2f}s"
                )
            else:
                # Recompute if the files are missing.
                _log_progress(
                    f"phase2 episode file missing, recomputing task={args.task_name} episode={episode_idx}"
                )
                phase2_rows = smooth_phase2_episode_rows(
                    rows,
                    model_type=args.model_type,
                    savgol_window=int(args.savgol_window),
                    savgol_polyorder=int(args.savgol_polyorder),
                )
                phase2_record = write_phase2_episode(
                    output_root=phase2_root,
                    task_name=args.task_name,
                    episode_idx=int(episode_idx),
                    phase1_metrics_jsonl=str(phase1_record["metrics_jsonl"]),
                    rows=phase2_rows,
                    save_plots=bool(args.save_plots),
                )
                phase2_records.append(phase2_record)
                _log_progress(
                    f"phase2 episode done task={args.task_name} episode={episode_idx} "
                    f"elapsed={time.perf_counter() - episode_start:.2f}s"
                )
        else:
            phase2_rows = smooth_phase2_episode_rows(
                rows,
                model_type=args.model_type,
                savgol_window=int(args.savgol_window),
                savgol_polyorder=int(args.savgol_polyorder),
            )
            phase2_record = write_phase2_episode(
                output_root=phase2_root,
                task_name=args.task_name,
                episode_idx=int(episode_idx),
                phase1_metrics_jsonl=str(phase1_record["metrics_jsonl"]),
                rows=phase2_rows,
                save_plots=bool(args.save_plots),
            )
            phase2_records.append(phase2_record)
            _log_progress(
                f"phase2 episode done task={args.task_name} episode={episode_idx} "
                f"elapsed={time.perf_counter() - episode_start:.2f}s"
            )

        write_phase2_manifest(
            output_root=phase2_root,
            task_name=args.task_name,
            model_type=args.model_type,
            repo_root=str(current_repo_root),
            phase1_root=phase1_root,
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
            episode_records=phase2_records,
        )

        if args.end_phase < 3:
            continue

        # ---------------- phase3: segmentation ----------------
        if args.resume and int(episode_idx) in existing_phase3_by_idx:
            phase3_record = existing_phase3_by_idx[int(episode_idx)]
            rows_jsonl = phase3_root / str(phase3_record["rows_jsonl"])
            # Recheck file existence.
            if rows_jsonl.exists():
                phase3_rows = read_jsonl(rows_jsonl)
                episode_success = bool(phase3_record.get("episode_success", True))
                phase3_records.append(phase3_record)
                _log_progress(
                    f"phase3 episode skipped (resume) task={args.task_name} episode={episode_idx} "
                    f"success={episode_success} elapsed={time.perf_counter() - episode_start:.2f}s"
                )
            else:
                # Recompute if the files are missing.
                _log_progress(
                    f"phase3 episode file missing, recomputing task={args.task_name} episode={episode_idx}"
                )
                phase3_rows, segments_payload, episode_success = segment_phase3_episode_rows(
                    phase2_rows,
                    savgol_window=int(args.savgol_window),
                    savgol_polyorder=int(args.savgol_polyorder),
                    peak_prominence=float(args.peak_prominence),
                    peak_distance_ratio=float(args.peak_distance_ratio),
                    min_rise_amplitude=float(args.min_rise_amplitude),
                    success_threshold=float(args.success_threshold),
                )
                phase3_record = write_phase3_episode(
                    output_root=phase3_root,
                    task_name=args.task_name,
                    episode_idx=int(episode_idx),
                    phase2_rows_jsonl=str(phase2_record["rows_jsonl"]),
                    rows=phase3_rows,
                    segments_payload=segments_payload,
                    episode_success=bool(episode_success),
                    save_plots=bool(args.save_plots),
                )
                phase3_records.append(phase3_record)
                _log_progress(
                    f"phase3 episode done task={args.task_name} episode={episode_idx} "
                    f"success={episode_success} elapsed={time.perf_counter() - episode_start:.2f}s"
                )
        else:
            phase3_rows, segments_payload, episode_success = segment_phase3_episode_rows(
                phase2_rows,
                savgol_window=int(args.savgol_window),
                savgol_polyorder=int(args.savgol_polyorder),
                peak_prominence=float(args.peak_prominence),
                peak_distance_ratio=float(args.peak_distance_ratio),
                min_rise_amplitude=float(args.min_rise_amplitude),
                success_threshold=float(args.success_threshold),
            )
            phase3_record = write_phase3_episode(
                output_root=phase3_root,
                task_name=args.task_name,
                episode_idx=int(episode_idx),
                phase2_rows_jsonl=str(phase2_record["rows_jsonl"]),
                rows=phase3_rows,
                segments_payload=segments_payload,
                episode_success=bool(episode_success),
                save_plots=bool(args.save_plots),
            )
            phase3_records.append(phase3_record)
            _log_progress(
                f"phase3 episode done task={args.task_name} episode={episode_idx} "
                f"success={episode_success} elapsed={time.perf_counter() - episode_start:.2f}s"
            )

        write_phase3_manifest(
            output_root=phase3_root,
            task_name=args.task_name,
            model_type=args.model_type,
            repo_root=str(current_repo_root),
            phase2_root=phase2_root,
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
            peak_prominence=float(args.peak_prominence),
            peak_distance_ratio=float(args.peak_distance_ratio),
            min_rise_amplitude=float(args.min_rise_amplitude),
            success_threshold=float(args.success_threshold),
            episode_records=phase3_records,
        )

    # ---------------- phase4: persistence (write back to parquet), after phases 1-3 finish for all episodes ----------------
    if args.end_phase >= 4:
        if export_dataset_root is None or phase4_metadata is None:
            raise RuntimeError("Phase4 context was not initialized")

        phase4_start = time.perf_counter()
        _log_progress(f"phase4 start task={args.task_name} (after all episodes phase1-3 completed)")

        # Run phase 4 for each completed episode.
        for episode_idx in episode_indices:
            episode_idx_int = int(episode_idx)
            # Get episode information from the phase 3 record.
            phase3_record = None
            for rec in phase3_records:
                rec_episode_idx_val = rec.get("episode_idx")
                if rec_episode_idx_val is not None:
                    try:
                        if int(rec_episode_idx_val) == episode_idx_int:  # type: ignore
                            phase3_record = rec
                            break
                    except (ValueError, TypeError):
                        continue

            if phase3_record is None:
                raise RuntimeError(f"Phase3 record not found for episode {episode_idx_int}")

            # Read phase 3 results.
            rows_jsonl = phase3_root / str(phase3_record["rows_jsonl"])
            phase3_rows = read_jsonl(rows_jsonl)

            if args.resume and episode_idx_int in existing_phase4_by_idx:
                phase4_record = existing_phase4_by_idx[episode_idx_int]
                phase4_records.append(phase4_record)
                rewritten_parquets.add(Path(str(phase4_record["source_parquet"])))
                _log_progress(
                    f"phase4 episode skipped (resume) task={args.task_name} episode={episode_idx}"
                )
            else:
                phase4_record, rewritten_src_parquet = rewrite_phase4_episode(
                    source_repo_root=current_repo_root,
                    export_dataset_root=export_dataset_root,
                    metadata=phase4_metadata,
                    episode_idx=episode_idx_int,
                    rows=phase3_rows,
                    value_column=args.value_column,
                )
                rewritten_parquets.add(rewritten_src_parquet)
                phase4_records.append(phase4_record)
                _log_progress(
                    f"phase4 episode done task={args.task_name} episode={episode_idx}"
                )

        # Copy untouched parquet files.
        _copy_untouched_parquet_files(current_repo_root, export_dataset_root, rewritten_parquets)
        
        # Update the metadata files.
        _update_meta_info(export_dataset_root, args.value_column, phase4_metadata)
        _update_meta_stats(export_dataset_root, args.value_column)
        
        # Write the final phase 4 manifest.
        write_phase4_manifest(
            output_root=phase4_root,
            task_name=args.task_name,
            source_repo_root=current_repo_root,
            phase3_root=phase3_root,
            export_dataset_root=export_dataset_root,
            value_column=args.value_column,
            episode_records=phase4_records,
        )
        _log_progress(f"phase4 done task={args.task_name} elapsed={time.perf_counter() - phase4_start:.2f}s")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    pipeline_start = time.perf_counter()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.start_phase < 0 or args.end_phase > 4 or args.start_phase > args.end_phase:
        raise ValueError(f"Invalid phase range: start_phase={args.start_phase}, end_phase={args.end_phase}")

    current_repo_root = Path(args.input_root).expanduser().resolve()
    phase0_manifest = None
    if args.source_mode == "hdf5":
        if args.start_phase == 0:
            phase0_root = output_root / "phase0_convert"
            phase0_manifest_path = phase0_root / "manifest.json"
            # Check the manifest to determine whether the phase 0 output directory exists.
            if args.resume and phase0_manifest_path.exists():
                _log_progress(f"phase0 skipped (resume) task={args.task_name}")
                phase0_manifest = read_json(phase0_manifest_path)
                current_repo_root = Path(phase0_manifest["output_dataset_root"]).resolve()
            else:
                phase0_start = time.perf_counter()
                _log_progress(f"phase0 start task={args.task_name} input={current_repo_root}")
                hdf5_dir = current_repo_root
                hdf5_paths = sorted(hdf5_dir.glob("*.hdf5"))
                if not hdf5_paths:
                    raise ValueError(f"No HDF5 files found under task root: {hdf5_dir}")
                phase0_manifest = convert_task_hdf5_to_lerobot(
                    task_name=args.task_name,
                    hdf5_paths=hdf5_paths,
                    output_root=phase0_root,
                    mode=args.mode,
                    fps=args.fps,
                    robot_type=args.robot_type,
                    overwrite=not args.resume,  # Do not overwrite existing outputs when resuming.
                    max_episodes_per_task=args.max_episodes_per_task,
                    num_workers=args.num_workers_per_task,
                )
                current_repo_root = Path(phase0_manifest["output_dataset_root"]).resolve()
                write_json(phase0_root / "manifest.json", phase0_manifest)
                _log_progress(f"phase0 done task={args.task_name} elapsed={time.perf_counter() - phase0_start:.2f}s")
        else:
            current_repo_root = output_root / "phase0_convert" / args.task_name
    elif args.start_phase == 0:
        args.start_phase = 1

    if args.end_phase == 0:
        write_json(
            output_root / "pipeline_manifest.json",
            {
                "task_name": args.task_name,
                "pipeline_mode": args.pipeline_mode,
                "source_mode": args.source_mode,
                "model_type": args.model_type,
                "input_root": str(Path(args.input_root).expanduser().resolve()),
                "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
                "output_root": str(output_root),
                "start_phase": int(args.start_phase),
                "end_phase": int(args.end_phase),
                "phase0_manifest": None if phase0_manifest is None else str((output_root / "phase0_convert" / "manifest.json")),
                "phase1_manifest": None,
                "phase2_manifest": None,
                "phase3_manifest": None,
                "phase4_manifest": None,
            },
        )
        _log_progress(f"pipeline done task={args.task_name} elapsed={time.perf_counter() - pipeline_start:.2f}s")
        return 0

    if args.pipeline_mode == "episode":
        _run_episode_pipeline(args, current_repo_root, output_root)
    else:
        _run_stage_pipeline(args, current_repo_root, output_root)

    write_json(
        output_root / "pipeline_manifest.json",
        {
            "task_name": args.task_name,
            "pipeline_mode": args.pipeline_mode,
            "source_mode": args.source_mode,
            "model_type": args.model_type,
            "input_root": str(Path(args.input_root).expanduser().resolve()),
            "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
            "output_root": str(output_root),
            "start_phase": int(args.start_phase),
            "end_phase": int(args.end_phase),
            "phase0_manifest": None if phase0_manifest is None else str((output_root / "phase0_convert" / "manifest.json")),
            "phase1_manifest": str(output_root / "phase1_eval_raw" / "manifest.json") if (output_root / "phase1_eval_raw" / "manifest.json").exists() else None,
            "phase2_manifest": str(output_root / "phase2_smooth" / "manifest.json") if (output_root / "phase2_smooth" / "manifest.json").exists() else None,
            "phase3_manifest": str(output_root / "phase3_segment" / "manifest.json") if (output_root / "phase3_segment" / "manifest.json").exists() else None,
            "phase4_manifest": str(output_root / "phase4_rewrite" / "manifest.json") if (output_root / "phase4_rewrite" / "manifest.json").exists() else None,
        },
    )
    _log_progress(f"pipeline done task={args.task_name} elapsed={time.perf_counter() - pipeline_start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
