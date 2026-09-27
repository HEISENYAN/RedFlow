from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Optional

from openpi.qwenvl.eval._artifacts import read_json
from openpi.qwenvl.eval._artifacts import read_jsonl
from openpi.qwenvl.eval._artifacts import write_csv
from openpi.qwenvl.eval._artifacts import write_json
from openpi.qwenvl.eval._artifacts import write_jsonl
from openpi.qwenvl.eval._curve_ops import detect_peak_valley_pairs
from openpi.qwenvl.eval._curve_ops import reconstruct_relabel_curve
from openpi.qwenvl.eval._curve_ops import resolve_episode_success


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: segment smoothed progress curves.")
    parser.add_argument("--phase2_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--savgol_window", default=11, type=int)
    parser.add_argument("--savgol_polyorder", default=2, type=int)
    parser.add_argument("--peak_prominence", default=0.08, type=float)
    parser.add_argument("--peak_distance_ratio", default=0.05, type=float)
    parser.add_argument("--min_rise_amplitude", default=0.10, type=float)
    parser.add_argument("--success_threshold", default=0.98, type=float)
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
    plt.plot(x_values, [float(row["segmented_value"]) for row in rows], label="segmented_value", linewidth=1.8)
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


def segment_phase3_episode_rows(
    rows: list[dict[str, Any]],
    *,
    savgol_window: int,
    savgol_polyorder: int,
    peak_prominence: float,
    peak_distance_ratio: float,
    min_rise_amplitude: float,
    success_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    progress01_raw = [float(row["progress01_raw"]) for row in rows]
    detection = detect_peak_valley_pairs(
        progress01_raw,
        savgol_window=int(savgol_window),
        savgol_polyorder=int(savgol_polyorder),
        peak_prominence=float(peak_prominence),
        peak_distance_ratio=float(peak_distance_ratio),
        min_rise_amplitude=float(min_rise_amplitude),
    )
    episode_success = resolve_episode_success(rows, success_threshold=float(success_threshold))
    segmented_values, realized_pairs = reconstruct_relabel_curve(
        detection.raw_progress01,
        detection,
        episode_success=episode_success,
    )
    output_rows = []
    for row, segmented_value in zip(rows, segmented_values.tolist(), strict=True):
        enriched = dict(row)
        enriched["segmented_value"] = float(segmented_value)
        output_rows.append(enriched)
    segments_payload = {
        "peak_indices": detection.peak_indices.tolist(),
        "valley_indices": detection.valley_indices.tolist(),
        "pairs": [asdict(pair) for pair in detection.pairs],
        "realized_pairs": [asdict(pair) for pair in realized_pairs],
    }
    return output_rows, segments_payload, bool(episode_success)


def write_phase3_episode(
    *,
    output_root: Path,
    task_name: str,
    episode_idx: int,
    phase2_rows_jsonl: str,
    rows: list[dict[str, Any]],
    segments_payload: dict[str, Any],
    episode_success: bool,
    save_plots: bool,
) -> dict[str, Any]:
    stem = f"episode_{int(episode_idx):06d}"
    rows_jsonl = output_root / "episodes" / f"{stem}.jsonl"
    rows_csv = output_root / "episodes" / f"{stem}.csv"
    segments_json = output_root / "segments" / f"{stem}.json"
    plot_path = output_root / "plots" / f"{stem}.png"
    write_jsonl(rows_jsonl, rows)
    write_csv(rows_csv, rows)
    write_json(
        segments_json,
        {
            "episode_idx": int(episode_idx),
            "episode_success": bool(episode_success),
            **segments_payload,
        },
    )
    if save_plots:
        _save_plot(plot_path, rows, title=f"{task_name} segmented episode {episode_idx}")
    return {
        "episode_idx": int(episode_idx),
        "phase2_rows_jsonl": phase2_rows_jsonl,
        "rows_jsonl": str(rows_jsonl.relative_to(output_root)),
        "rows_csv": str(rows_csv.relative_to(output_root)),
        "segments_json": str(segments_json.relative_to(output_root)),
        "plot": str(plot_path.relative_to(output_root)) if save_plots else None,
        "num_frames": int(len(rows)),
        "episode_success": bool(episode_success),
    }


def write_phase3_manifest(
    *,
    output_root: Path,
    task_name: str,
    model_type: str,
    repo_root: str | Path,
    phase2_root: str | Path,
    savgol_window: int,
    savgol_polyorder: int,
    peak_prominence: float,
    peak_distance_ratio: float,
    min_rise_amplitude: float,
    success_threshold: float,
    episode_records: list[dict[str, Any]],
):
    write_json(
        output_root / "manifest.json",
        {
            "phase": 3,
            "task_name": task_name,
            "model_type": str(model_type),
            "repo_root": str(repo_root),
            "phase2_root": str(Path(phase2_root).expanduser().resolve()),
            "params": {
                "savgol_window": int(savgol_window),
                "savgol_polyorder": int(savgol_polyorder),
                "peak_prominence": float(peak_prominence),
                "peak_distance_ratio": float(peak_distance_ratio),
                "min_rise_amplitude": float(min_rise_amplitude),
                "success_threshold": float(success_threshold),
            },
            "episodes": episode_records,
        },
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    phase2_root = Path(args.phase2_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    phase2_manifest = read_json(phase2_root / "manifest.json")
    manifest_episodes = []
    for episode in phase2_manifest["episodes"]:
        episode_idx = int(episode["episode_idx"])
        rows = read_jsonl(phase2_root / episode["rows_jsonl"])
        phase3_rows, segments_payload, episode_success = segment_phase3_episode_rows(
            rows,
            savgol_window=int(args.savgol_window),
            savgol_polyorder=int(args.savgol_polyorder),
            peak_prominence=float(args.peak_prominence),
            peak_distance_ratio=float(args.peak_distance_ratio),
            min_rise_amplitude=float(args.min_rise_amplitude),
            success_threshold=float(args.success_threshold),
        )
        manifest_episodes.append(
            write_phase3_episode(
                output_root=output_root,
                task_name=args.task_name,
                episode_idx=episode_idx,
                phase2_rows_jsonl=episode["rows_jsonl"],
                rows=phase3_rows,
                segments_payload=segments_payload,
                episode_success=episode_success,
                save_plots=bool(args.save_plots),
            )
        )
    write_phase3_manifest(
        output_root=output_root,
        task_name=args.task_name,
        model_type=str(phase2_manifest.get("model_type", "pi06")),
        repo_root=phase2_manifest["repo_root"],
        phase2_root=phase2_root,
        savgol_window=int(args.savgol_window),
        savgol_polyorder=int(args.savgol_polyorder),
        peak_prominence=float(args.peak_prominence),
        peak_distance_ratio=float(args.peak_distance_ratio),
        min_rise_amplitude=float(args.min_rise_amplitude),
        success_threshold=float(args.success_threshold),
        episode_records=manifest_episodes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
