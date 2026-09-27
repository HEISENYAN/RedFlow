from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from typing import Optional

from openpi.qwenvl.eval._artifacts import read_json
from openpi.qwenvl.eval._artifacts import read_jsonl
from openpi.qwenvl.eval._artifacts import write_csv
from openpi.qwenvl.eval._artifacts import write_json
from openpi.qwenvl.eval._artifacts import write_jsonl
from openpi.qwenvl.eval._curve_ops import map_absolute_to_progress01
from openpi.qwenvl.eval._curve_ops import smooth_progress_curve


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: smooth raw value curves.")
    parser.add_argument("--phase1_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--savgol_window", default=11, type=int)
    parser.add_argument("--savgol_polyorder", default=2, type=int)
    parser.add_argument("--save_plots", default=True, type=lambda value: str(value).lower() in {"1", "true", "yes", "on"})
    return parser.parse_args(argv)


def _save_plot(path: Path, rows: list[dict[str, float]], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = [int(row["dataset_frame_index"]) for row in rows]
    plt.figure(figsize=(12, 4.5))
    plt.plot(x_values, [float(row["progress01_raw"]) for row in rows], label="progress01_raw", linewidth=1.1)
    plt.plot(x_values, [float(row["progress01_smooth"]) for row in rows], label="progress01_smooth", linewidth=1.4)
    plt.xlabel("frame_index")
    plt.ylabel("progress")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def smooth_phase2_episode_rows(
    rows: list[dict[str, Any]],
    *,
    model_type: str,
    savgol_window: int,
    savgol_polyorder: int,
) -> list[dict[str, Any]]:
    del model_type
    absolute_values = [float(row["absolute_value"]) for row in rows]
    progress01_raw = map_absolute_to_progress01(absolute_values)
    progress01_smooth = smooth_progress_curve(
        progress01_raw,
        window_length=int(savgol_window),
        polyorder=int(savgol_polyorder),
    )
    output_rows = []
    for row, raw_value, smooth_value in zip(rows, progress01_raw.tolist(), progress01_smooth.tolist(), strict=True):
        enriched = dict(row)
        enriched["progress01_raw"] = float(raw_value)
        enriched["progress01_smooth"] = float(smooth_value)
        output_rows.append(enriched)
    return output_rows


def write_phase2_episode(
    *,
    output_root: Path,
    task_name: str,
    episode_idx: int,
    phase1_metrics_jsonl: str,
    rows: list[dict[str, Any]],
    save_plots: bool,
) -> dict[str, Any]:
    stem = f"episode_{int(episode_idx):06d}"
    rows_jsonl = output_root / "episodes" / f"{stem}.jsonl"
    rows_csv = output_root / "episodes" / f"{stem}.csv"
    plot_path = output_root / "plots" / f"{stem}.png"
    write_jsonl(rows_jsonl, rows)
    write_csv(rows_csv, rows)
    if save_plots:
        _save_plot(plot_path, rows, title=f"{task_name} smooth episode {episode_idx}")
    return {
        "episode_idx": int(episode_idx),
        "phase1_metrics_jsonl": phase1_metrics_jsonl,
        "rows_jsonl": str(rows_jsonl.relative_to(output_root)),
        "rows_csv": str(rows_csv.relative_to(output_root)),
        "plot": str(plot_path.relative_to(output_root)) if save_plots else None,
        "num_frames": int(len(rows)),
    }


def write_phase2_manifest(
    *,
    output_root: Path,
    task_name: str,
    model_type: str,
    repo_root: str | Path,
    phase1_root: str | Path,
    savgol_window: int,
    savgol_polyorder: int,
    episode_records: list[dict[str, Any]],
):
    write_json(
        output_root / "manifest.json",
        {
            "phase": 2,
            "task_name": task_name,
            "model_type": str(model_type),
            "repo_root": str(repo_root),
            "phase1_root": str(Path(phase1_root).expanduser().resolve()),
            "params": {
                "savgol_window": int(savgol_window),
                "savgol_polyorder": int(savgol_polyorder),
            },
            "episodes": episode_records,
        },
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    phase1_root = Path(args.phase1_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    phase1_manifest = read_json(phase1_root / "manifest.json")
    manifest_episodes = []
    for episode in phase1_manifest["episodes"]:
        episode_idx = int(episode["episode_idx"])
        rows = read_jsonl(phase1_root / episode["metrics_jsonl"])
        output_rows = smooth_phase2_episode_rows(
            rows,
            model_type=str(phase1_manifest.get("model_type", "pi06")),
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
        )
        manifest_episodes.append(
            write_phase2_episode(
                output_root=output_root,
                task_name=args.task_name,
                episode_idx=episode_idx,
                phase1_metrics_jsonl=episode["metrics_jsonl"],
                rows=output_rows,
                save_plots=bool(args.save_plots),
            )
        )
    write_phase2_manifest(
        output_root=output_root,
        task_name=args.task_name,
        model_type=str(phase1_manifest.get("model_type", "pi06")),
        repo_root=phase1_manifest["repo_root"],
        phase1_root=phase1_root,
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        episode_records=manifest_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
