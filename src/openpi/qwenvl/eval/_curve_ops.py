from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class PeakValleyPair:
    valley_idx: int
    peak_idx: int
    peak_value: float
    valley_value: float
    rise: float
    source: str


@dataclass(frozen=True)
class DetectionResult:
    raw_progress01: np.ndarray
    smooth_progress01: np.ndarray
    peak_indices: np.ndarray
    valley_indices: np.ndarray
    pairs: tuple[PeakValleyPair, ...]


def clip01(values: np.ndarray | float) -> np.ndarray | float:
    if isinstance(values, np.ndarray):
        return np.clip(values.astype(np.float32), 0.0, 1.0)
    if isinstance(values, (list, tuple)):
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return float(max(0.0, min(1.0, float(values))))


def to_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "success", "yes", "y"}:
            return True
        if lowered in {"0", "false", "fail", "failure", "no", "n"}:
            return False
    return bool(value)


def map_absolute_to_progress01(absolute_values: np.ndarray) -> np.ndarray:
    absolute_values = np.asarray(absolute_values, dtype=np.float32)
    return clip01(absolute_values + 1.0)


def smooth_progress_curve(progress01: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    progress01 = np.asarray(progress01, dtype=np.float32)
    length = int(len(progress01))
    if length < 3:
        return progress01.copy()
    window = min(int(window_length), length)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return progress01.copy()
    poly = min(int(polyorder), window - 1)
    if poly < 1:
        return progress01.copy()
    smoothed = savgol_filter(progress01, window_length=window, polyorder=poly, mode="interp")
    return np.clip(smoothed.astype(np.float32), 0.0, 1.0)


def detect_peak_valley_pairs(
    progress01: np.ndarray,
    *,
    savgol_window: int = 11,
    savgol_polyorder: int = 2,
    peak_prominence: float = 0.08,
    peak_distance_ratio: float = 0.05,
    min_rise_amplitude: float = 0.10,
) -> DetectionResult:
    raw = np.asarray(clip01(progress01), dtype=np.float32)
    smooth = smooth_progress_curve(raw, window_length=savgol_window, polyorder=savgol_polyorder)
    distance = max(1, int(round(len(raw) * float(peak_distance_ratio))))
    peak_indices, _ = find_peaks(smooth, prominence=float(peak_prominence), distance=distance)
    valley_indices, _ = find_peaks(-smooth, prominence=float(peak_prominence), distance=distance)
    sorted_valleys = sorted({0, *[int(index) for index in valley_indices.tolist()]})
    pairs = []
    for peak_idx in peak_indices.tolist():
        peak_idx = int(peak_idx)
        if peak_idx <= 0:
            continue
        candidates = [idx for idx in sorted_valleys if idx < peak_idx]
        valley_idx = candidates[-1] if candidates else 0
        rise = float(raw[peak_idx] - raw[valley_idx])
        if rise < float(min_rise_amplitude):
            continue
        pairs.append(
            PeakValleyPair(
                valley_idx=valley_idx,
                peak_idx=peak_idx,
                peak_value=float(raw[peak_idx]),
                valley_value=float(raw[valley_idx]),
                rise=rise,
                source="peak",
            )
        )
    return DetectionResult(
        raw_progress01=raw,
        smooth_progress01=smooth,
        peak_indices=np.asarray(peak_indices, dtype=np.int64),
        valley_indices=np.asarray(valley_indices, dtype=np.int64),
        pairs=tuple(pairs),
    )


def _build_single_rise_segment(length: int, pair: PeakValleyPair, step_penalty: float) -> np.ndarray:
    segment = np.zeros(int(length), dtype=np.float32)
    valley_idx = int(pair.valley_idx)
    peak_idx = int(pair.peak_idx)
    peak_value = float(pair.peak_value)
    if peak_idx <= valley_idx:
        segment[peak_idx] = peak_value
        return segment
    valley_value = max(0.0, peak_value - (peak_idx - valley_idx) * float(step_penalty))
    segment[valley_idx : peak_idx + 1] = np.linspace(
        valley_value,
        peak_value,
        num=(peak_idx - valley_idx + 1),
        dtype=np.float32,
    )
    return segment


def reconstruct_relabel_curve(
    progress01: np.ndarray,
    detection: DetectionResult,
    *,
    episode_success: bool,
) -> tuple[np.ndarray, list[PeakValleyPair]]:
    raw = np.asarray(clip01(progress01), dtype=np.float32)
    length = int(len(raw))
    step_penalty = 1.0 / max(length - 1, 1)
    envelope = np.zeros(length, dtype=np.float32)
    realized_pairs = list(detection.pairs)
    valleys_for_terminal = sorted({0, *[int(index) for index in detection.valley_indices.tolist()]})
    final_anchor = 1.0 if episode_success else float(raw[-1])
    terminal_candidates = [idx for idx in valleys_for_terminal if idx < length - 1]
    terminal_valley_idx = terminal_candidates[-1] if terminal_candidates else 0
    realized_pairs.append(
        PeakValleyPair(
            valley_idx=int(terminal_valley_idx),
            peak_idx=int(length - 1),
            peak_value=float(final_anchor),
            valley_value=float(raw[terminal_valley_idx]),
            rise=float(final_anchor - raw[terminal_valley_idx]),
            source="terminal_success" if episode_success else "terminal_failure",
        )
    )
    for pair in realized_pairs:
        envelope = np.maximum(envelope, _build_single_rise_segment(length, pair, step_penalty))
    return envelope.astype(np.float32), realized_pairs


def resolve_episode_success(rows: list[dict[str, Any]], success_threshold: float) -> bool:
    for key in ("result", "episode_success", "success"):
        for row in rows:
            value = row.get(key)
            if value is not None:
                return to_bool(value)
    return float(rows[-1]["progress01_raw"]) >= float(success_threshold)
