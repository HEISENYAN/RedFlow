from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from openpi.qwenvl.eval._artifacts import write_json


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-level multi-GPU scheduler for the strict QwenVL eval pipeline.")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--pipeline_mode", default="episode", choices=("episode", "stage"))
    parser.add_argument("--model_type", default="pi06", choices=("pi06", "pair"))
    parser.add_argument("--source_mode", required=True, choices=("hdf5", "lerobot"))
    parser.add_argument("--input_root", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
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


def _task_input_root(source_mode: str, input_root: Path, task_name: str) -> Path:
    task_root = input_root / task_name
    if not task_root.exists():
        raise FileNotFoundError(f"Missing task input for task={task_name}: {task_root}")
    return task_root


def _build_pipeline_cmd(args: argparse.Namespace, task_name: str, task_input_root: Path, task_output_root: Path) -> list[str]:
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise FileNotFoundError("`uv` was not found in PATH.")
    cmd = [
        uv_bin,
        "run",
        "--python",
        str((Path.cwd() / ".venv" / "bin" / "python").resolve()),
        str((Path.cwd() / "src" / "openpi" / "qwenvl" / "eval" / "run_pipeline.py").resolve()),
        "--pipeline_mode",
        args.pipeline_mode,
        "--model_type",
        args.model_type,
        "--source_mode",
        args.source_mode,
        "--input_root",
        str(task_input_root),
        "--checkpoint_dir",
        str(Path(args.checkpoint_dir).expanduser().resolve()),
        "--task_name",
        task_name,
        "--output_root",
        str(task_output_root),
        "--start_phase",
        str(args.start_phase),
        "--end_phase",
        str(args.end_phase),
        "--camera_names",
        args.camera_names,
        "--mode",
        args.mode,
        "--fps",
        str(args.fps),
        "--robot_type",
        args.robot_type,
        "--batch_size",
        str(args.batch_size),
        "--stream_chunk_size",
        str(args.stream_chunk_size),
        "--frame_interval",
        str(args.frame_interval),
        "--relative_interval",
        str(args.relative_interval),
        "--save_plots",
        str(bool(args.save_plots)).lower(),
        "--save_videos",
        str(bool(args.save_videos)).lower(),
        "--video_fps",
        str(args.video_fps),
        "--attn_implementation",
        args.attn_implementation,
        "--bf16",
        str(bool(args.bf16)).lower(),
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
        "--value_column",
        args.value_column,
    ]
    if args.max_episodes_per_task is not None:
        cmd += ["--max_episodes_per_task", str(args.max_episodes_per_task)]
    if args.max_episodes is not None:
        cmd += ["--max_episodes", str(args.max_episodes)]
    if args.episode_indices:
        cmd += ["--episode_indices", *[str(value) for value in args.episode_indices]]
    if args.prompt is not None:
        cmd += ["--prompt", args.prompt]
    cmd += ["--num_workers_per_task", str(args.num_workers_per_task)]
    if args.final_dir is not None:
        cmd += ["--final_dir", str(args.final_dir)]
    cmd += ["--resume", str(bool(args.resume)).lower()]
    return cmd


def _launch_task(
    *,
    args: argparse.Namespace,
    input_root: Path,
    logs_root: Path,
    gpu_id: str,
    task_name: str,
    task_output_root: Path,
):
    task_input_root = _task_input_root(args.source_mode, input_root, task_name)
    log_path = logs_root / f"{task_name}_gpu_{gpu_id}.log"
    cmd = _build_pipeline_cmd(args, task_name, task_input_root, task_output_root)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_fh = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    print(f"[launch] gpu={gpu_id} task={task_name} log={log_path}", flush=True)
    return proc, log_fh, log_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    logs_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    assignments: list[list[str]] = [[] for _ in args.gpus]
    for task_index, task_name in enumerate(args.tasks):
        assignments[task_index % len(args.gpus)].append(task_name)

    gpu_labels = [str(gpu_id) for gpu_id in args.gpus]
    print(
        f"[scheduler] tasks={len(args.tasks)} gpus={','.join(gpu_labels)} "
        f"input={input_root} output={output_root}",
        flush=True,
    )

    workers: list[dict[str, object]] = []
    for worker_index, gpu_id in enumerate(gpu_labels):
        queue = list(assignments[worker_index])
        state: dict[str, object] = {
            "gpu_id": gpu_id,
            "queue": queue,
            "proc": None,
            "log_fh": None,
            "log_path": None,
            "task_name": None,
            "task_output_root": None,
        }
        if queue:
            task_name = queue.pop(0)
            task_output_root = output_root / task_name
            proc, log_fh, log_path = _launch_task(
                args=args,
                input_root=input_root,
                logs_root=logs_root,
                gpu_id=gpu_id,
                task_name=task_name,
                task_output_root=task_output_root,
            )
            state.update(
                {
                    "proc": proc,
                    "log_fh": log_fh,
                    "log_path": log_path,
                    "task_name": task_name,
                    "task_output_root": task_output_root,
                }
            )
        workers.append(state)

    summary_tasks = []
    total_tasks = len(args.tasks)
    completed_tasks = 0
    last_status_time = 0.0
    while completed_tasks < total_tasks:
        made_progress = False
        active_labels: list[str] = []
        for state in workers:
            proc = state["proc"]
            task_name = state["task_name"]
            gpu_id = state["gpu_id"]
            if proc is None or task_name is None:
                continue
            active_labels.append(f"{task_name}@{gpu_id}")
            return_code = proc.poll()
            if return_code is None:
                continue

            made_progress = True
            log_fh = state["log_fh"]
            log_path = state["log_path"]
            task_output_root = state["task_output_root"]
            if log_fh is not None:
                log_fh.close()
            if return_code != 0:
                raise RuntimeError(f"Task {task_name} failed on GPU {gpu_id}. See log: {log_path}")
            pipeline_manifest = Path(task_output_root) / "pipeline_manifest.json"
            if not pipeline_manifest.exists():
                raise FileNotFoundError(f"Task {task_name} completed without pipeline manifest: {pipeline_manifest}")
            completed_tasks += 1
            print(
                f"[done] {completed_tasks}/{total_tasks} gpu={gpu_id} task={task_name} "
                f"manifest={pipeline_manifest}",
                flush=True,
            )
            summary_tasks.append(
                {
                    "task_name": task_name,
                    "gpu_id": str(gpu_id),
                    "log_path": str(log_path),
                    "pipeline_manifest": str(pipeline_manifest),
                }
            )

            queue = state["queue"]
            if queue:
                next_task = queue.pop(0)
                next_output_root = output_root / next_task
                next_proc, next_log_fh, next_log_path = _launch_task(
                    args=args,
                    input_root=input_root,
                    logs_root=logs_root,
                    gpu_id=str(gpu_id),
                    task_name=next_task,
                    task_output_root=next_output_root,
                )
                state.update(
                    {
                        "proc": next_proc,
                        "log_fh": next_log_fh,
                        "log_path": next_log_path,
                        "task_name": next_task,
                        "task_output_root": next_output_root,
                    }
                )
            else:
                state.update(
                    {
                        "proc": None,
                        "log_fh": None,
                        "log_path": None,
                        "task_name": None,
                        "task_output_root": None,
                    }
                )

        now = time.time()
        if not made_progress and now - last_status_time >= 15:
            active = ", ".join(active_labels) if active_labels else "idle"
            print(f"[running] {completed_tasks}/{total_tasks} active={active}", flush=True)
            last_status_time = now
        if completed_tasks < total_tasks:
            time.sleep(1)
    write_json(
        output_root / "scheduler_summary.json",
        {
            "tasks": summary_tasks,
            "gpus": [str(gpu_id) for gpu_id in args.gpus],
            "pipeline_mode": args.pipeline_mode,
            "source_mode": args.source_mode,
            "model_type": args.model_type,
            "input_root": str(input_root),
            "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
            "output_root": str(output_root),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        },
    )
    print(json.dumps({"tasks": summary_tasks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
