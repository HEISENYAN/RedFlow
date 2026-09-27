from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.qwenvl.eval._artifacts import read_json
from openpi.qwenvl.eval._artifacts import read_jsonl
from openpi.qwenvl.eval._artifacts import write_json
from openpi.qwenvl.eval._lerobot_io import load_lerobot_metadata
from openpi.qwenvl.eval._lerobot_io import resolve_episode_local_row_range
from openpi.qwenvl.eval._lerobot_io import resolve_parquet_path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: rewrite segmented value curves back to LeRobot parquet.")
    parser.add_argument("--phase3_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--value_column", default="value", type=str)
    parser.add_argument("--final_dir", default=None, type=str, help="Final output directory. If provided, dataset will be copied to final_dir/task_name")
    return parser.parse_args(argv)


def _hard_copy(src: Path, dst: Path):
    """Always copy files rather than symlinking so the dataset can be moved independently."""
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _copy_dataset_complete(src_root: Path, dst_root: Path):
    """Copy the entire dataset, including everything in the data directory."""
    dst_root.mkdir(parents=True, exist_ok=True)
    for item in src_root.iterdir():
        target = dst_root / item.name
        if target.exists():
            continue
        _hard_copy(item, target)


def _copy_untouched_parquet_files(src_root: Path, dst_root: Path, rewritten_parquets: set[Path]):
    """Copy untouched parquet files. Since the full dataset has already been copied, this mainly ensures those files exist."""
    for src_parquet in src_root.glob("data/**/*.parquet"):
        if src_parquet in rewritten_parquets:
            continue
        dst_parquet = dst_root / src_parquet.relative_to(src_root)
        # Skip destination files that already exist from the full copy.
        if dst_parquet.exists():
            continue
        # Copy missing destination files; this should be rare after the full copy.
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        _hard_copy(src_parquet, dst_parquet)


def _upsert_value_column(
    base_parquet: Path,
    dst_parquet: Path,
    value_column: str,
    values: list[float],
    *,
    row_start: int,
    row_end: int,
):
    table = pq.read_table(base_parquet)
    if int(row_end) - int(row_start) != len(values):
        raise ValueError(
            f"Value count mismatch for {base_parquet}: values={len(values)} row_span={int(row_end) - int(row_start)}"
        )
    if value_column in table.column_names:
        merged_values = table[value_column].to_pylist()
    else:
        merged_values = [None] * table.num_rows
    for offset, value in enumerate(values):
        merged_values[int(row_start) + offset] = float(value)
    column = pa.array(merged_values, type=pa.float32())
    if value_column in table.column_names:
        table = table.set_column(table.column_names.index(value_column), value_column, column)
    else:
        table = table.append_column(value_column, column)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst_parquet)


def init_phase4_context(source_repo_root: str | Path, output_root: str | Path, task_name: str | None = None, final_dir: str | Path | None = None):
    source_root = Path(source_repo_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    
    # If final_dir is provided, use final_dir/task_name as the output directory.
    if final_dir is not None:
        final_path = Path(final_dir).expanduser().resolve()
        if task_name is None:
            raise ValueError("task_name is required when final_dir is provided")
        export_dataset_root = final_path / task_name
    else:
        export_dataset_root = output_path / f"{source_root.name}_rewritten"
    
    # Copy the entire dataset, including the data directory, to produce a complete LeRobot dataset.
    _copy_dataset_complete(source_root, export_dataset_root)
    metadata = load_lerobot_metadata(source_root)
    return export_dataset_root, metadata


def rewrite_phase4_episode(
    *,
    source_repo_root: str | Path,
    export_dataset_root: str | Path,
    metadata,
    episode_idx: int,
    rows: list[dict[str, Any]],
    value_column: str,
) -> tuple[dict[str, Any], Path]:
    source_root = Path(source_repo_root).expanduser().resolve()
    export_root = Path(export_dataset_root).expanduser().resolve()
    src_parquet = resolve_parquet_path(source_root, metadata, int(episode_idx))
    row_start, row_end = resolve_episode_local_row_range(metadata, int(episode_idx))
    values = [float(row["segmented_value"]) for row in rows]
    if int(row_end) - int(row_start) != len(values):
        raise ValueError(
            f"Episode {episode_idx} row count mismatch: segmented_rows={len(values)} "
            f"episode_rows={int(row_end) - int(row_start)}"
        )
    dst_parquet = export_root / src_parquet.relative_to(source_root)
    base_parquet = dst_parquet if dst_parquet.exists() else src_parquet
    _upsert_value_column(
        base_parquet,
        dst_parquet,
        value_column,
        values,
        row_start=row_start,
        row_end=row_end,
    )
    return (
        {
            "episode_idx": int(episode_idx),
            "source_parquet": str(src_parquet),
            "output_parquet": str(dst_parquet),
            "num_frames": int(len(values)),
            "row_start": int(row_start),
            "row_end": int(row_end),
            "value_column": str(value_column),
        },
        src_parquet,
    )


def _update_meta_info(export_dataset_root: Path, value_column: str, metadata) -> None:
    """Update meta/info.json with the feature definition for the value column."""
    info_path = export_dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")
    
    info = read_json(info_path)
    
    # Add the feature definition for the value column.
    if "features" not in info:
        info["features"] = {}
    
    fps = getattr(metadata, "fps", 30)
    info["features"][value_column] = {
        "dtype": "float32",
        "shape": [1],
        "names": [value_column],
        "fps": int(fps),
    }
    
    write_json(info_path, info)


def _compute_value_stats(export_dataset_root: Path, value_column: str) -> dict[str, Any]:
    """Compute statistics for the value column."""
    all_values = []
    
    # Collect values from the value column across all parquet files.
    for parquet_path in export_dataset_root.glob("data/**/*.parquet"):
        table = pq.read_table(parquet_path)
        if value_column in table.column_names:
            values = table[value_column].to_pylist()
            # Filter out None values.
            all_values.extend([v for v in values if v is not None])
    
    if not all_values:
        return {}
    
    values_array = np.array(all_values, dtype=np.float32)
    
    return {
        "min": [float(np.min(values_array))],
        "max": [float(np.max(values_array))],
        "mean": [float(np.mean(values_array))],
        "std": [float(np.std(values_array))],
        "count": [len(all_values)],
    }


def _update_meta_stats(export_dataset_root: Path, value_column: str) -> None:
    """Update meta/stats.json with statistics for the value column."""
    stats_path = export_dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"meta/stats.json not found: {stats_path}")
    
    stats = read_json(stats_path)
    
    # Compute and add statistics for the value column.
    value_stats = _compute_value_stats(export_dataset_root, value_column)
    if value_stats:
        stats[value_column] = value_stats
    
    write_json(stats_path, stats)


def write_phase4_manifest(
    *,
    output_root: Path,
    task_name: str,
    source_repo_root: str | Path,
    phase3_root: str | Path,
    export_dataset_root: str | Path,
    value_column: str,
    episode_records: list[dict[str, Any]],
):
    write_json(
        output_root / "manifest.json",
        {
            "phase": 4,
            "task_name": task_name,
            "source_repo_root": str(Path(source_repo_root).expanduser().resolve()),
            "phase3_root": str(Path(phase3_root).expanduser().resolve()),
            "export_dataset_root": str(Path(export_dataset_root).expanduser().resolve()),
            "value_column": str(value_column),
            "episodes": episode_records,
        },
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    phase3_root = Path(args.phase3_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    phase3_manifest = read_json(phase3_root / "manifest.json")
    source_repo_root = Path(phase3_manifest["repo_root"]).resolve()
    export_dataset_root, metadata = init_phase4_context(
        source_repo_root, output_root, task_name=args.task_name, final_dir=args.final_dir
    )

    rewritten_records = []
    rewritten_parquets: set[Path] = set()
    for episode in phase3_manifest["episodes"]:
        episode_idx = int(episode["episode_idx"])
        rows = read_jsonl(phase3_root / episode["rows_jsonl"])
        phase4_record, rewritten_src_parquet = rewrite_phase4_episode(
            source_repo_root=source_repo_root,
            export_dataset_root=export_dataset_root,
            metadata=metadata,
            episode_idx=episode_idx,
            rows=rows,
            value_column=args.value_column,
        )
        rewritten_records.append(phase4_record)
        rewritten_parquets.add(rewritten_src_parquet)
    _copy_untouched_parquet_files(source_repo_root, export_dataset_root, rewritten_parquets)
    
    # Update the metadata files.
    _update_meta_info(export_dataset_root, args.value_column, metadata)
    _update_meta_stats(export_dataset_root, args.value_column)
    
    write_phase4_manifest(
        output_root=output_root,
        task_name=args.task_name,
        source_repo_root=source_repo_root,
        phase3_root=phase3_root,
        export_dataset_root=export_dataset_root,
        value_column=args.value_column,
        episode_records=rewritten_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
