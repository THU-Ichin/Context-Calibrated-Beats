#!/usr/bin/env python3
"""Analyse beats and local tempo with Beat This! on one or more music files.

For every input file, this script writes the official raw and overlap-fused:
  * framewise beat/downbeat logits and probabilities;
  * beat CSVs with raw and smoothed local BPM;
  * WAV files with audible clicks at the detected beats;
  * tempo and probability comparison plots;
  * a CSV summary and a JSON report of possible tempo-change regions.

The raw local BPM is exactly 60 / (time between adjacent detected beats).
The smoothed value is intended for visualisation and change detection only.
It corrects isolated likely missed/double beats and applies a short median filter;
the raw values are always retained in the output CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.special import expit


def configure_plot_fonts() -> str | None:
    """Select an installed font that covers Japanese song titles."""
    candidates = (
        "BIZ UDPGothic",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans JP",
        "MS Gothic",
        "Noto Sans CJK JP",
        "Hiragino Sans",
        "IPAexGothic",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in installed), None)
    if selected is not None:
        plt.rcParams["font.family"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


PLOT_FONT = configure_plot_fonts()


@dataclass
class BeatResult:
    method: str
    beat_times: np.ndarray
    reported_bpm: float | None = None
    downbeat_times: np.ndarray | None = None
    note: str = ""


@dataclass
class FramePredictions:
    fps: float
    raw_beat_logits: np.ndarray
    raw_downbeat_logits: np.ndarray
    fused_beat_logits: np.ndarray
    fused_downbeat_logits: np.ndarray
    window_seconds: float
    hop_seconds: float
    overlap_windows: int


@dataclass
class GridDecoding:
    interval_midpoints: np.ndarray
    interval_durations: np.ndarray
    base_bpm: np.ndarray
    selected_scale: np.ndarray
    normalized_bpm: np.ndarray
    confidence: np.ndarray
    segment_id: np.ndarray


@dataclass
class TempoRegion:
    start_seconds: float
    end_seconds: float
    median_bpm: float
    difference_from_main_bpm: float
    beat_intervals: int


@dataclass
class ActivitySegment:
    segment_id: int
    start_seconds: float
    end_seconds: float
    no_beat: bool
    source: str = "default"
    note: str = ""


@dataclass
class InferenceMetadata:
    fps: float
    window_seconds: float
    hop_seconds: float
    overlap_windows: int
    sample_rate: int
    duration_seconds: float
    audio_path: str


class BeatThisEstimator:
    """Run the official baseline plus overlap-weighted frame inference."""

    FPS = 50.0
    WINDOW_FRAMES = 1500
    BORDER_FRAMES = 6

    def __init__(self, device: str, checkpoint: str, hop_seconds: float = 10.0):
        try:
            import torch
            import torch.nn.functional as torch_functional
            from beat_this.inference import Audio2Frames
            from beat_this.model.postprocessor import Postprocessor
        except ImportError as exc:
            raise RuntimeError(
                "Beat This! is not installed. Run: python -m pip install beat-this"
            ) from exc
        if not 0 < hop_seconds < self.WINDOW_FRAMES / self.FPS:
            raise ValueError("Beat This! overlap hop must be between 0 and 30 seconds")
        self.torch = torch
        self.torch_functional = torch_functional
        self.frame_model = Audio2Frames(checkpoint_path=checkpoint, device=device)
        self.postprocessor = Postprocessor(type="minimal", fps=int(self.FPS))
        self.hop_frames = max(1, round(hop_seconds * self.FPS))

    def _overlap_fused_logits(self, spect) -> tuple[object, object, int]:
        """Infer shifted 30-second windows and Hann-average their logits."""
        torch = self.torch
        border = self.BORDER_FRAMES
        full_frames = int(spect.shape[0])
        padded = self.torch_functional.pad(spect, (0, 0, border, border))
        padded_frames = int(padded.shape[0])

        if padded_frames <= self.WINDOW_FRAMES:
            starts = [0]
        else:
            last_start = padded_frames - self.WINDOW_FRAMES
            starts = list(range(0, last_start + 1, self.hop_frames))
            if starts[-1] != last_start:
                starts.append(last_start)

        beat_sum = torch.zeros(padded_frames, device=self.frame_model.device)
        downbeat_sum = torch.zeros_like(beat_sum)
        weight_sum = torch.zeros_like(beat_sum)
        with torch.inference_mode():
            with torch.autocast(
                enabled=self.frame_model.float16,
                device_type=self.frame_model.device.type,
            ):
                for start in starts:
                    chunk = padded[start : start + self.WINDOW_FRAMES]
                    prediction = self.frame_model.model(chunk.unsqueeze(0))
                    beat = prediction["beat"][0].float()
                    downbeat = prediction["downbeat"][0].float()
                    weights = torch.hann_window(
                        len(beat),
                        periodic=False,
                        dtype=beat.dtype,
                        device=beat.device,
                    ).clamp_min(1e-3)
                    end = start + len(beat)
                    beat_sum[start:end] += beat * weights
                    downbeat_sum[start:end] += downbeat * weights
                    weight_sum[start:end] += weights

        if torch.any(weight_sum <= 0):
            raise RuntimeError("Overlap fusion left one or more frames uncovered")
        fused_beat = beat_sum / weight_sum
        fused_downbeat = downbeat_sum / weight_sum
        return (
            fused_beat[border : border + full_frames],
            fused_downbeat[border : border + full_frames],
            len(starts),
        )

    @staticmethod
    def _numpy(tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy()

    def __call__(
        self, y: np.ndarray, sr: int
    ) -> tuple[list[BeatResult], FramePredictions]:
        # Decode once with librosa, then reuse the same spectrogram for the
        # official hard-splice baseline and the shifted-overlap inference.
        spect = self.frame_model.signal2spect(y, sr)
        raw_beat_logits, raw_downbeat_logits = self.frame_model.spect2frames(spect)
        fused_beat_logits, fused_downbeat_logits, overlap_windows = (
            self._overlap_fused_logits(spect)
        )

        raw_beats, raw_downbeats = self.postprocessor(
            raw_beat_logits, raw_downbeat_logits
        )
        fused_beats, fused_downbeats = self.postprocessor(
            fused_beat_logits, fused_downbeat_logits
        )
        results = [
            BeatResult(
                method="beat-this-raw",
                beat_times=np.asarray(raw_beats, dtype=float),
                downbeat_times=np.asarray(raw_downbeats, dtype=float),
                note="Official 30-second keep-first chunk aggregation",
            ),
            BeatResult(
                method="beat-this-fused",
                beat_times=np.asarray(fused_beats, dtype=float),
                downbeat_times=np.asarray(fused_downbeats, dtype=float),
                note=(
                    "30-second shifted windows with Hann-weighted logit fusion "
                    f"and {self.hop_frames / self.FPS:g}-second hop"
                ),
            ),
        ]
        frames = FramePredictions(
            fps=self.FPS,
            raw_beat_logits=self._numpy(raw_beat_logits),
            raw_downbeat_logits=self._numpy(raw_downbeat_logits),
            fused_beat_logits=self._numpy(fused_beat_logits),
            fused_downbeat_logits=self._numpy(fused_downbeat_logits),
            window_seconds=self.WINDOW_FRAMES / self.FPS,
            hop_seconds=self.hop_frames / self.FPS,
            overlap_windows=overlap_windows,
        )
        return results, frames


def local_tempo(beat_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw and robust local BPM between adjacent Beat This! beats."""
    times = np.asarray(beat_times, dtype=float)
    if len(times) < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    intervals = np.diff(times)
    midpoints = (times[:-1] + times[1:]) / 2.0
    valid = intervals > 1e-5
    intervals = intervals[valid]
    midpoints = midpoints[valid]
    raw = 60.0 / intervals
    if not len(raw):
        return midpoints, raw, raw.copy()

    # A short local median supplies context. Isolated values very close to
    # half/double that context usually indicate a missed or extra detected beat.
    window = min(5, len(raw) if len(raw) % 2 == 1 else len(raw) - 1)
    window = max(1, window)
    context = median_filter(raw, size=window, mode="nearest")
    corrected = raw.copy()
    ratio = np.divide(raw, context, out=np.ones_like(raw), where=context > 0)
    corrected[(ratio > 1.75) & (ratio < 2.25)] /= 2.0
    corrected[(ratio > 0.44) & (ratio < 0.58)] *= 2.0
    smooth = median_filter(corrected, size=window, mode="nearest")
    return midpoints, raw, smooth


NORMALIZED_BPM_MIN = 120.0
NORMALIZED_BPM_MAX = 240.0
GRID_SCALES = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0], dtype=float)


def decode_grid_scales(base_bpm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose a metrical scale path whose BPM stays in [120, 240)."""
    bpm = np.asarray(base_bpm, dtype=float)
    if not len(bpm):
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    normalized = bpm[:, None] * GRID_SCALES[None, :]
    valid = (normalized >= NORMALIZED_BPM_MIN) & (normalized < NORMALIZED_BPM_MAX)
    distance = np.where(
        normalized < NORMALIZED_BPM_MIN,
        np.log2(NORMALIZED_BPM_MIN / np.maximum(normalized, 1e-6)),
        np.where(
            normalized >= NORMALIZED_BPM_MAX,
            np.log2(normalized / NORMALIZED_BPM_MAX),
            0.0,
        ),
    )
    # Staying outside the canonical range is much more expensive than changing
    # metrical level. Continuity still breaks ties and suppresses weak flicker.
    emission = np.where(valid, 0.0, 25.0 + 80.0 * distance**2)
    n_frames, n_states = emission.shape
    costs = np.full((n_frames, n_states), np.inf)
    backpointers = np.zeros((n_frames, n_states), dtype=int)
    costs[0] = emission[0]
    for index in range(1, n_frames):
        previous_normalized = normalized[index - 1]
        for state in range(n_states):
            tempo_jump = np.abs(
                np.log2(
                    np.maximum(normalized[index, state], 1e-6)
                    / np.maximum(previous_normalized, 1e-6)
                )
            )
            scale_jump = np.abs(np.log2(GRID_SCALES[state] / GRID_SCALES))
            transition = 3.0 * np.minimum(tempo_jump, 1.0) + 0.6 * scale_jump
            candidates = costs[index - 1] + transition
            best_previous = int(np.argmin(candidates))
            costs[index, state] = emission[index, state] + candidates[best_previous]
            backpointers[index, state] = best_previous

    states = np.zeros(n_frames, dtype=int)
    states[-1] = int(np.argmin(costs[-1]))
    for index in range(n_frames - 1, 0, -1):
        states[index - 1] = backpointers[index, states[index]]

    sorted_emission = np.sort(emission, axis=1)
    margin = sorted_emission[:, 1] - sorted_emission[:, 0]
    confidence = 1.0 - np.exp(-margin / 10.0)
    return states, confidence


def _frame_index(time_seconds: float, frames: FramePredictions) -> int:
    return int(
        np.clip(
            round(time_seconds * frames.fps),
            0,
            len(frames.fused_beat_logits) - 1,
        )
    )


def _build_normalized_grid_segment(
    fused_result: BeatResult,
    frames: FramePredictions,
    snap_radius_seconds: float = 0.08,
) -> tuple[BeatResult, GridDecoding]:
    """Build a normalized grid for one contiguous beat-active segment."""
    beat_times = np.asarray(fused_result.beat_times, dtype=float)
    midpoints, _, base_bpm = local_tempo(beat_times)
    if len(beat_times) < 2 or not len(base_bpm):
        empty = np.asarray([], dtype=float)
        return (
            BeatResult(
                method="beat-this-normalized",
                beat_times=beat_times.copy(),
                downbeat_times=np.asarray([], dtype=float),
                note="Insufficient beats for metrical-level normalization",
            ),
            GridDecoding(empty, empty, empty, empty, empty, empty, empty),
        )

    states, confidence = decode_grid_scales(base_bpm)
    selected_scale = GRID_SCALES[states]
    normalized_bpm = base_bpm * selected_scale
    segment_id = np.r_[0, np.cumsum(states[1:] != states[:-1])].astype(int)
    beat_probability = expit(frames.fused_beat_logits)
    downbeat_probability = expit(frames.fused_downbeat_logits)
    radius_frames = max(1, round(snap_radius_seconds * frames.fps))

    def evidence(time_seconds: float) -> float:
        index = _frame_index(time_seconds, frames)
        return float(
            beat_probability[index] + 0.2 * downbeat_probability[index]
        )

    def snap_inserted_beat(time_seconds: float) -> tuple[float, float]:
        center = _frame_index(time_seconds, frames)
        left = max(0, center - radius_frames)
        right = min(len(beat_probability), center + radius_frames + 1)
        best = left + int(np.argmax(frames.fused_beat_logits[left:right]))
        if beat_probability[best] >= 0.5:
            return best / frames.fps, evidence(best / frames.fps)
        return time_seconds, evidence(time_seconds)

    candidates: list[tuple[float, float]] = []
    segment_starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    segment_ends = np.r_[segment_starts[1:], len(states)]
    for start, end in zip(segment_starts, segment_ends):
        scale = selected_scale[start]
        if scale >= 1.0:
            multiplier = int(round(scale))
            for interval in range(int(start), int(end)):
                left_time = beat_times[interval]
                right_time = beat_times[interval + 1]
                candidates.append((left_time, evidence(left_time)))
                for subdivision in range(1, multiplier):
                    target = left_time + (right_time - left_time) * (
                        subdivision / multiplier
                    )
                    candidates.append(snap_inserted_beat(target))
            boundary = beat_times[int(end)]
            candidates.append((boundary, evidence(boundary)))
        else:
            stride = int(round(1.0 / scale))
            indices = np.arange(int(start), int(end) + 1)
            best_phase = 0
            best_score = -np.inf
            for phase in range(stride):
                selected = indices[(indices - int(start) - phase) % stride == 0]
                if not len(selected):
                    continue
                frame_indices = np.asarray(
                    [_frame_index(beat_times[index], frames) for index in selected]
                )
                score = float(
                    np.mean(beat_probability[frame_indices])
                    + 0.35 * np.mean(downbeat_probability[frame_indices])
                )
                if score > best_score:
                    best_score = score
                    best_phase = phase
            selected = indices[
                (indices - int(start) - best_phase) % stride == 0
            ]
            for index in selected:
                time_seconds = beat_times[index]
                candidates.append((time_seconds, evidence(time_seconds)))

    # Preserve the outer extent, then collapse duplicate boundary candidates.
    candidates.extend(
        [
            (beat_times[0], evidence(beat_times[0])),
            (beat_times[-1], evidence(beat_times[-1])),
        ]
    )
    candidates.sort(key=lambda item: item[0])
    deduplicated: list[tuple[float, float]] = []
    for candidate in candidates:
        if deduplicated and candidate[0] - deduplicated[-1][0] < 0.08:
            if candidate[1] > deduplicated[-1][1]:
                deduplicated[-1] = candidate
        else:
            deduplicated.append(candidate)

    # Enforce the canonical range on the final click grid, including transition
    # boundaries. Resolve over-dense conflicts by evidence, then fill gaps.
    minimum_period = 60.0 / NORMALIZED_BPM_MAX
    maximum_period = 60.0 / NORMALIZED_BPM_MIN
    constrained = deduplicated.copy()
    while len(constrained) >= 2:
        intervals = np.diff([item[0] for item in constrained])
        conflicts = np.flatnonzero(intervals <= minimum_period + 1e-6)
        if not len(conflicts):
            break
        left = int(conflicts[0])
        right = left + 1
        if left == 0:
            remove = right
        elif right == len(constrained) - 1:
            remove = left
        else:
            remove = left if constrained[left][1] < constrained[right][1] else right
        constrained.pop(remove)

    regularized: list[tuple[float, float]] = []
    for index, item in enumerate(constrained[:-1]):
        regularized.append(item)
        next_item = constrained[index + 1]
        gap = next_item[0] - item[0]
        subdivisions = max(1, int(math.ceil(gap / maximum_period)))
        for subdivision in range(1, subdivisions):
            time_seconds = item[0] + gap * subdivision / subdivisions
            regularized.append((time_seconds, evidence(time_seconds)))
    if constrained:
        regularized.append(constrained[-1])
    corrected_beats = np.asarray([item[0] for item in regularized], dtype=float)

    source_downbeats = np.asarray(
        fused_result.downbeat_times
        if fused_result.downbeat_times is not None
        else [],
        dtype=float,
    )
    corrected_downbeats: list[float] = []
    for time_seconds in corrected_beats:
        index = _frame_index(time_seconds, frames)
        has_source_downbeat = bool(
            source_downbeats.size
            and np.min(np.abs(source_downbeats - time_seconds)) <= 0.07
        )
        if has_source_downbeat or downbeat_probability[index] >= 0.5:
            corrected_downbeats.append(time_seconds)

    result = BeatResult(
        method="beat-this-normalized",
        beat_times=corrected_beats,
        downbeat_times=np.asarray(corrected_downbeats, dtype=float),
        note=(
            "Offline [120, 240) BPM normalization over "
            "0.25x/0.5x/1x/2x/4x metrical grids"
        ),
    )
    decoding = GridDecoding(
        interval_midpoints=midpoints,
        interval_durations=np.diff(beat_times),
        base_bpm=base_bpm,
        selected_scale=selected_scale,
        normalized_bpm=normalized_bpm,
        confidence=confidence,
        segment_id=segment_id,
    )
    return result, decoding


def build_normalized_grid(
    fused_result: BeatResult,
    frames: FramePredictions,
    active_ranges: list[tuple[float, float]],
    snap_radius_seconds: float = 0.08,
) -> tuple[BeatResult, GridDecoding]:
    """Normalize each beat-active range independently without bridging gaps."""
    normalized_results: list[BeatResult] = []
    decodings: list[GridDecoding] = []
    next_grid_segment = 0
    beat_times = np.asarray(fused_result.beat_times, dtype=float)
    downbeat_times = np.asarray(
        fused_result.downbeat_times
        if fused_result.downbeat_times is not None
        else [],
        dtype=float,
    )

    for range_index, (start, end) in enumerate(active_ranges):
        is_last = range_index == len(active_ranges) - 1
        beat_mask = (beat_times >= start) & (
            (beat_times <= end) if is_last else (beat_times < end)
        )
        segment_beats = beat_times[beat_mask]
        if len(segment_beats) < 2:
            continue
        downbeat_mask = (downbeat_times >= start) & (
            (downbeat_times <= end) if is_last else (downbeat_times < end)
        )
        segment_result = BeatResult(
            method=fused_result.method,
            beat_times=segment_beats,
            downbeat_times=downbeat_times[downbeat_mask],
            note=fused_result.note,
        )
        normalized, decoding = _build_normalized_grid_segment(
            segment_result,
            frames,
            snap_radius_seconds=snap_radius_seconds,
        )
        if len(decoding.segment_id):
            decoding.segment_id = decoding.segment_id + next_grid_segment
            next_grid_segment = int(decoding.segment_id[-1]) + 1
        normalized_results.append(normalized)
        decodings.append(decoding)

    empty = np.asarray([], dtype=float)
    if not normalized_results:
        return (
            BeatResult(
                method="beat-this-normalized",
                beat_times=empty,
                downbeat_times=empty,
                note="No beat-active segment contained enough beats",
            ),
            GridDecoding(empty, empty, empty, empty, empty, empty, empty),
        )

    corrected_beats = np.unique(
        np.concatenate([result.beat_times for result in normalized_results])
    )
    corrected_downbeats = np.unique(
        np.concatenate(
            [
                result.downbeat_times
                for result in normalized_results
                if result.downbeat_times is not None
                and len(result.downbeat_times)
            ]
        )
        if any(
            result.downbeat_times is not None and len(result.downbeat_times)
            for result in normalized_results
        )
        else empty
    )
    result = BeatResult(
        method="beat-this-normalized",
        beat_times=corrected_beats,
        downbeat_times=corrected_downbeats,
        note=(
            "Offline [120, 240) BPM normalization over "
            "0.25x/0.5x/1x/2x/4x metrical grids within beat-active segments"
        ),
    )
    decoding = GridDecoding(
        interval_midpoints=np.concatenate(
            [item.interval_midpoints for item in decodings]
        ),
        interval_durations=np.concatenate(
            [item.interval_durations for item in decodings]
        ),
        base_bpm=np.concatenate([item.base_bpm for item in decodings]),
        selected_scale=np.concatenate(
            [item.selected_scale for item in decodings]
        ),
        normalized_bpm=np.concatenate(
            [item.normalized_bpm for item in decodings]
        ),
        confidence=np.concatenate([item.confidence for item in decodings]),
        segment_id=np.concatenate([item.segment_id for item in decodings]),
    )
    return result, decoding


def modal_tempo(
    midpoints: np.ndarray,
    bpm: np.ndarray,
    min_bpm: float,
    max_bpm: float,
    weights: np.ndarray | None = None,
) -> float:
    """Find the time-weighted dominant tempo rather than merely the median."""
    valid = np.isfinite(bpm) & (bpm >= min_bpm) & (bpm <= max_bpm)
    if not np.any(valid):
        return math.nan
    values = bpm[valid]
    valid_times = midpoints[valid]
    if weights is not None:
        histogram_weights = np.asarray(weights, dtype=float)[valid]
    elif len(valid_times) > 1:
        weights = np.gradient(valid_times)
        histogram_weights = np.clip(weights, 1e-3, None)
    else:
        histogram_weights = np.ones_like(values)

    edges = np.arange(math.floor(min_bpm), math.ceil(max_bpm) + 1.0, 1.0)
    hist, _ = np.histogram(values, bins=edges, weights=histogram_weights)
    hist = gaussian_filter1d(hist.astype(float), sigma=1.5)
    index = int(np.argmax(hist))
    return float((edges[index] + edges[index + 1]) / 2.0)


def detect_change_regions(
    midpoints: np.ndarray,
    bpm: np.ndarray,
    main_bpm: float,
    relative_threshold: float,
    absolute_threshold: float,
    min_intervals: int,
    min_duration: float,
) -> list[TempoRegion]:
    """Group sustained intervals whose tempo differs materially from the mode."""
    if not len(bpm) or not np.isfinite(main_bpm):
        return []
    threshold = max(absolute_threshold, main_bpm * relative_threshold)
    changed = np.abs(bpm - main_bpm) >= threshold

    # Fill a one-interval hole to avoid splitting a genuine changed section.
    if len(changed) >= 3:
        holes = (~changed[1:-1]) & changed[:-2] & changed[2:]
        changed[np.flatnonzero(holes) + 1] = True

    regions: list[TempoRegion] = []
    start: int | None = None
    for i, flag in enumerate(np.r_[changed, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i
            count = end - start
            left_step = (midpoints[start] - midpoints[start - 1]) if start > 0 else 0.0
            right_step = (midpoints[end] - midpoints[end - 1]) if end < len(midpoints) else 0.0
            region_start = max(0.0, midpoints[start] - left_step / 2.0)
            region_end = midpoints[end - 1] + right_step / 2.0
            duration = region_end - region_start
            if count >= min_intervals and duration >= min_duration:
                region_bpm = float(np.median(bpm[start:end]))
                regions.append(
                    TempoRegion(
                        start_seconds=round(region_start, 3),
                        end_seconds=round(region_end, 3),
                        median_bpm=round(region_bpm, 2),
                        difference_from_main_bpm=round(region_bpm - main_bpm, 2),
                        beat_intervals=count,
                    )
                )
            start = None
    return regions


def analyse_result_from_ranges(
    result: BeatResult,
    ranges: list[tuple[float, float]],
    duration: float,
    args: argparse.Namespace,
    audio_path: Path,
) -> dict:
    """Analyse only active ranges and insert NaNs so plots do not bridge gaps."""
    pieces: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    beat_times = np.asarray(result.beat_times, dtype=float)
    for index, (start, end) in enumerate(ranges):
        is_last = index == len(ranges) - 1
        mask = (beat_times >= start) & (
            (beat_times <= end) if is_last else (beat_times < end)
        )
        segment_times = beat_times[mask]
        midpoints, raw_bpm, smooth_bpm = local_tempo(segment_times)
        if len(midpoints):
            pieces.append(
                (midpoints, raw_bpm, smooth_bpm, np.diff(segment_times))
            )

    if pieces:
        modal_midpoints = np.concatenate([piece[0] for piece in pieces])
        modal_bpm = np.concatenate([piece[2] for piece in pieces])
        modal_weights = np.concatenate([piece[3] for piece in pieces])
    else:
        modal_midpoints = np.asarray([], dtype=float)
        modal_bpm = np.asarray([], dtype=float)
        modal_weights = np.asarray([], dtype=float)
    main_bpm = modal_tempo(
        modal_midpoints,
        modal_bpm,
        args.min_bpm,
        args.max_bpm,
        weights=modal_weights,
    )
    regions: list[TempoRegion] = []
    for midpoints, _, smooth_bpm, _ in pieces:
        regions.extend(
            detect_change_regions(
                midpoints,
                smooth_bpm,
                main_bpm,
                relative_threshold=args.change_ratio,
                absolute_threshold=args.change_bpm,
                min_intervals=args.min_change_beats,
                min_duration=args.min_change_seconds,
            )
        )

    plot_midpoints: list[float] = []
    plot_raw: list[float] = []
    plot_smooth: list[float] = []
    for piece_index, (midpoints, raw_bpm, smooth_bpm, _) in enumerate(pieces):
        if piece_index:
            plot_midpoints.append(float("nan"))
            plot_raw.append(float("nan"))
            plot_smooth.append(float("nan"))
        plot_midpoints.extend(midpoints)
        plot_raw.extend(raw_bpm)
        plot_smooth.extend(smooth_bpm)

    return {
        "result": filter_result_to_ranges(result, ranges),
        "midpoints": np.asarray(plot_midpoints, dtype=float),
        "raw_bpm": np.asarray(plot_raw, dtype=float),
        "smooth_bpm": np.asarray(plot_smooth, dtype=float),
        "main_bpm": main_bpm,
        "regions": regions,
        "duration": duration,
        "active_duration": sum(end - start for start, end in ranges),
        "no_beat_duration": duration - sum(end - start for start, end in ranges),
        "audio": audio_path,
    }


def safe_stem(path: Path) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem)
    return cleaned.strip(" ._") or "audio"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def ensure_segments_csv(path: Path, duration: float) -> None:
    """Create an editable default activity map without overwriting user edits."""
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "segment_id",
                "start_seconds",
                "end_seconds",
                "no_beat",
                "source",
                "note",
            ]
        )
        writer.writerow([0, "0.000000000", f"{duration:.9f}", 0, "default", ""])


def read_segments_csv(path: Path, duration: float) -> list[ActivitySegment]:
    """Read NO_BEAT annotations; true rows override uncovered/default rows."""
    segments: list[ActivitySegment] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
                no_beat = _parse_bool(row.get("no_beat", "0"))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid segment row {row_number} in {path.name}: {exc}"
                ) from exc
            if not (0.0 <= start < end <= duration + 1e-6):
                raise ValueError(
                    f"Segment row {row_number} must satisfy "
                    f"0 <= start < end <= {duration:.6f}"
                )
            segments.append(
                ActivitySegment(
                    segment_id=int(row.get("segment_id", len(segments))),
                    start_seconds=max(0.0, start),
                    end_seconds=min(duration, end),
                    no_beat=no_beat,
                    source=(row.get("source") or "user").strip(),
                    note=(row.get("note") or "").strip(),
                )
            )
    return segments


def no_beat_ranges(
    segments: list[ActivitySegment], duration: float
) -> list[tuple[float, float]]:
    """Return the merged union of user/model NO_BEAT ranges."""
    blocked = sorted(
        (
            max(0.0, item.start_seconds),
            min(duration, item.end_seconds),
        )
        for item in segments
        if item.no_beat and item.end_seconds > 0 and item.start_seconds < duration
    )
    merged: list[tuple[float, float]] = []
    for start, end in blocked:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def active_ranges(
    segments: list[ActivitySegment], duration: float
) -> list[tuple[float, float]]:
    """Return the complement of all NO_BEAT annotations."""
    active: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in no_beat_ranges(segments, duration):
        if start > cursor + 1e-9:
            active.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 1e-9:
        active.append((cursor, duration))
    return active


def filter_result_to_ranges(
    result: BeatResult, ranges: list[tuple[float, float]]
) -> BeatResult:
    """Keep beats/downbeats that fall within at least one active range."""
    beat_times = np.asarray(result.beat_times, dtype=float)
    downbeat_times = np.asarray(
        result.downbeat_times if result.downbeat_times is not None else [],
        dtype=float,
    )

    def keep_mask(times: np.ndarray) -> np.ndarray:
        mask = np.zeros(len(times), dtype=bool)
        for index, (start, end) in enumerate(ranges):
            is_last = index == len(ranges) - 1
            mask |= (times >= start) & ((times <= end) if is_last else (times < end))
        return mask

    return BeatResult(
        method=result.method,
        beat_times=beat_times[keep_mask(beat_times)],
        reported_bpm=result.reported_bpm,
        downbeat_times=downbeat_times[keep_mask(downbeat_times)],
        note=result.note,
    )


def write_inference_metadata_csv(path: Path, metadata: InferenceMetadata) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metadata)))
        writer.writeheader()
        writer.writerow(asdict(metadata))


def read_inference_metadata_csv(path: Path) -> InferenceMetadata:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"Inference metadata is empty: {path}")
    return InferenceMetadata(
        fps=float(row["fps"]),
        window_seconds=float(row["window_seconds"]),
        hop_seconds=float(row["hop_seconds"]),
        overlap_windows=int(row["overlap_windows"]),
        sample_rate=int(row["sample_rate"]),
        duration_seconds=float(row["duration_seconds"]),
        audio_path=row["audio_path"],
    )


def write_frame_predictions_csv(path: Path, frames: FramePredictions) -> None:
    lengths = {
        len(frames.raw_beat_logits),
        len(frames.raw_downbeat_logits),
        len(frames.fused_beat_logits),
        len(frames.fused_downbeat_logits),
    }
    if len(lengths) != 1:
        raise ValueError(f"Frame prediction lengths differ: {sorted(lengths)}")

    raw_beat_prob = expit(frames.raw_beat_logits)
    raw_downbeat_prob = expit(frames.raw_downbeat_logits)
    fused_beat_prob = expit(frames.fused_beat_logits)
    fused_downbeat_prob = expit(frames.fused_downbeat_logits)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "time_seconds",
                "raw_beat_logit",
                "raw_beat_probability",
                "raw_downbeat_logit",
                "raw_downbeat_probability",
                "fused_beat_logit",
                "fused_beat_probability",
                "fused_downbeat_logit",
                "fused_downbeat_probability",
            ]
        )
        for index in range(lengths.pop()):
            writer.writerow(
                [
                    index,
                    f"{index / frames.fps:.9f}",
                    f"{frames.raw_beat_logits[index]:.7f}",
                    f"{raw_beat_prob[index]:.7f}",
                    f"{frames.raw_downbeat_logits[index]:.7f}",
                    f"{raw_downbeat_prob[index]:.7f}",
                    f"{frames.fused_beat_logits[index]:.7f}",
                    f"{fused_beat_prob[index]:.7f}",
                    f"{frames.fused_downbeat_logits[index]:.7f}",
                    f"{fused_downbeat_prob[index]:.7f}",
                ]
            )


def read_frame_predictions_csv(
    path: Path, metadata: InferenceMetadata
) -> FramePredictions:
    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return FramePredictions(
        fps=metadata.fps,
        raw_beat_logits=np.asarray(
            [float(row["raw_beat_logit"]) for row in rows], dtype=float
        ),
        raw_downbeat_logits=np.asarray(
            [float(row["raw_downbeat_logit"]) for row in rows], dtype=float
        ),
        fused_beat_logits=np.asarray(
            [float(row["fused_beat_logit"]) for row in rows], dtype=float
        ),
        fused_downbeat_logits=np.asarray(
            [float(row["fused_downbeat_logit"]) for row in rows], dtype=float
        ),
        window_seconds=metadata.window_seconds,
        hop_seconds=metadata.hop_seconds,
        overlap_windows=metadata.overlap_windows,
    )


def write_grid_decisions_csv(path: Path, decoding: GridDecoding) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "interval_index",
                "interval_midpoint_seconds",
                "interval_duration_seconds",
                "base_smoothed_bpm",
                "selected_scale",
                "normalized_bpm",
                "scale_confidence",
                "segment_id",
            ]
        )
        for index in range(len(decoding.interval_midpoints)):
            writer.writerow(
                [
                    index,
                    f"{decoding.interval_midpoints[index]:.9f}",
                    f"{decoding.interval_durations[index]:.9f}",
                    f"{decoding.base_bpm[index]:.4f}",
                    f"{decoding.selected_scale[index]:g}",
                    f"{decoding.normalized_bpm[index]:.4f}",
                    f"{decoding.confidence[index]:.6f}",
                    int(decoding.segment_id[index]),
                ]
            )


def read_grid_decisions_csv(path: Path) -> GridDecoding:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return GridDecoding(
        interval_midpoints=np.asarray(
            [float(row["interval_midpoint_seconds"]) for row in rows], dtype=float
        ),
        interval_durations=np.asarray(
            [float(row["interval_duration_seconds"]) for row in rows], dtype=float
        ),
        base_bpm=np.asarray(
            [float(row["base_smoothed_bpm"]) for row in rows], dtype=float
        ),
        selected_scale=np.asarray(
            [float(row["selected_scale"]) for row in rows], dtype=float
        ),
        normalized_bpm=np.asarray(
            [float(row["normalized_bpm"]) for row in rows], dtype=float
        ),
        confidence=np.asarray(
            [float(row["scale_confidence"]) for row in rows], dtype=float
        ),
        segment_id=np.asarray(
            [int(row["segment_id"]) for row in rows], dtype=int
        ),
    )


def write_beats_csv(
    path: Path,
    result: BeatResult,
    midpoints: np.ndarray,
    raw_bpm: np.ndarray,
    smooth_bpm: np.ndarray,
    sample_rate: int,
    ranges: list[tuple[float, float]] | None = None,
) -> None:
    downbeats = np.asarray(result.downbeat_times if result.downbeat_times is not None else [])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "beat_index",
                "sample_index",
                "beat_time_seconds",
                "is_downbeat",
                "activity_segment_id",
                "is_no_beat",
                "interval_midpoint_seconds",
                "raw_local_bpm",
                "smoothed_local_bpm",
            ]
        )

        def activity_id(time_seconds: float) -> int:
            if ranges is None:
                return 0
            for range_index, (start, end) in enumerate(ranges):
                is_last = range_index == len(ranges) - 1
                if time_seconds >= start and (
                    time_seconds <= end if is_last else time_seconds < end
                ):
                    return range_index
            return -1

        for i, beat_time in enumerate(result.beat_times):
            is_downbeat = bool(
                downbeats.size and np.min(np.abs(downbeats - beat_time)) <= 0.05
            )
            current_activity = activity_id(float(beat_time))
            previous_activity = (
                activity_id(float(result.beat_times[i - 1])) if i else -1
            )
            has_active_interval = (
                i > 0
                and i - 1 < len(raw_bpm)
                and current_activity >= 0
                and current_activity == previous_activity
            )
            if not has_active_interval:
                row = [
                    i + 1,
                    int(round(beat_time * sample_rate)),
                    f"{beat_time:.9f}",
                    int(is_downbeat),
                    current_activity,
                    int(current_activity < 0),
                    "",
                    "",
                    "",
                ]
            else:
                row = [
                    i + 1,
                    int(round(beat_time * sample_rate)),
                    f"{beat_time:.9f}",
                    int(is_downbeat),
                    current_activity,
                    int(current_activity < 0),
                    f"{midpoints[i - 1]:.9f}",
                    f"{raw_bpm[i - 1]:.4f}",
                    f"{smooth_bpm[i - 1]:.4f}",
                ]
            writer.writerow(row)


def read_beats_csv(path: Path, method: str, note: str = "") -> BeatResult:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    beat_times = np.asarray(
        [float(row["beat_time_seconds"]) for row in rows], dtype=float
    )
    downbeat_times = np.asarray(
        [
            float(row["beat_time_seconds"])
            for row in rows
            if _parse_bool(row.get("is_downbeat", "0"))
        ],
        dtype=float,
    )
    return BeatResult(
        method=method,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        note=note,
    )


def write_click_track(path: Path, y: np.ndarray, sr: int, result: BeatResult) -> None:
    clicks = librosa.clicks(
        times=result.beat_times,
        sr=sr,
        click_freq=1200.0,
        click_duration=0.035,
        length=len(y),
    )
    if result.downbeat_times is not None and len(result.downbeat_times):
        clicks += 1.35 * librosa.clicks(
            times=result.downbeat_times,
            sr=sr,
            click_freq=2200.0,
            click_duration=0.045,
            length=len(y),
        )
    # Keep the music audible while ensuring clicks can be heard without clipping.
    mix = 0.1 * librosa.util.normalize(y) + 0.9 * librosa.util.normalize(clicks)
    mix = np.clip(mix, -1.0, 1.0)
    sf.write(path, mix, sr, subtype="PCM_16")


def write_plot(
    path: Path,
    title: str,
    analyses: list[dict],
    duration: float,
    blocked_ranges: list[tuple[float, float]],
) -> None:
    fig, axes = plt.subplots(
        len(analyses),
        1,
        figsize=(13, 3.4 * len(analyses)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, analysis in zip(axes, analyses):
        for blocked_index, (start, end) in enumerate(blocked_ranges):
            ax.axvspan(
                start,
                end,
                color="0.5",
                alpha=0.16,
                label="NO_BEAT" if blocked_index == 0 else None,
            )
        mids = analysis["midpoints"]
        raw = analysis["raw_bpm"]
        smooth = analysis["smooth_bpm"]
        main = analysis["main_bpm"]
        ax.scatter(mids, raw, s=9, alpha=0.22, label="raw 60 / beat interval")
        ax.plot(mids, smooth, linewidth=1.5, label="robust local BPM")
        if np.isfinite(main):
            ax.axhline(
                main,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"main {main:.1f}",
            )
        for region in analysis["regions"]:
            ax.axvspan(
                region.start_seconds,
                region.end_seconds,
                color="tab:red",
                alpha=0.12,
            )
        ax.set_ylabel("BPM")
        ax.set_title(analysis["result"].method)
        ax.set_xlim(0, duration)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_probability_plot(
    path: Path,
    title: str,
    frames: FramePredictions,
    duration: float,
    blocked_ranges: list[tuple[float, float]],
) -> None:
    times = np.arange(len(frames.raw_beat_logits), dtype=float) / frames.fps
    series = [
        (
            "Beat probability",
            expit(frames.raw_beat_logits),
            expit(frames.fused_beat_logits),
        ),
        (
            "Downbeat probability",
            expit(frames.raw_downbeat_logits),
            expit(frames.fused_downbeat_logits),
        ),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    for ax, (label, raw, fused) in zip(axes, series):
        for blocked_index, (start, end) in enumerate(blocked_ranges):
            ax.axvspan(
                start,
                end,
                color="0.5",
                alpha=0.16,
                label="NO_BEAT" if blocked_index == 0 else None,
            )
        ax.plot(times, raw, linewidth=0.65, alpha=0.65, label="official raw")
        ax.plot(times, fused, linewidth=0.8, alpha=0.8, label="overlap fused")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(label)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlim(0, duration)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_grid_plot(
    path: Path,
    title: str,
    decoding: GridDecoding,
    normalized_result: BeatResult,
    ranges: list[tuple[float, float]],
    blocked_ranges: list[tuple[float, float]],
    duration: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    for range_index, (start, end) in enumerate(ranges):
        mask = (decoding.interval_midpoints >= start) & (
            decoding.interval_midpoints < end
        )
        if np.any(mask):
            axes[0].plot(
                decoding.interval_midpoints[mask],
                decoding.base_bpm[mask],
                color="tab:blue",
                linewidth=1.0,
                alpha=0.65,
                label="fused base BPM" if range_index == 0 else None,
            )
            axes[0].plot(
                decoding.interval_midpoints[mask],
                decoding.normalized_bpm[mask],
                color="tab:orange",
                linewidth=1.3,
                label="grid decision BPM" if range_index == 0 else None,
            )
            axes[1].step(
                decoding.interval_midpoints[mask],
                decoding.selected_scale[mask],
                color="tab:blue",
                where="mid",
                linewidth=1.2,
            )
        beat_mask = (normalized_result.beat_times >= start) & (
            normalized_result.beat_times < end
        )
        midpoints, _, actual_bpm = local_tempo(
            normalized_result.beat_times[beat_mask]
        )
        if len(midpoints):
            axes[0].plot(
                midpoints,
                actual_bpm,
                color="tab:green",
                linewidth=1.0,
                linestyle="--",
                alpha=0.85,
                label="final actual BPM" if range_index == 0 else None,
            )
    for ax in axes:
        for blocked_index, (start, end) in enumerate(blocked_ranges):
            ax.axvspan(
                start,
                end,
                color="0.5",
                alpha=0.16,
                label="NO_BEAT" if blocked_index == 0 else None,
            )
    axes[0].axhspan(
        NORMALIZED_BPM_MIN,
        NORMALIZED_BPM_MAX,
        color="tab:green",
        alpha=0.08,
        label="target [120, 240)",
    )
    axes[0].set_ylabel("BPM")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_yscale("log", base=2)
    axes[1].set_yticks(GRID_SCALES, [f"{scale:g}x" for scale in GRID_SCALES])
    axes[1].set_ylabel("Selected grid")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_xlim(0, duration)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyse_file(
    audio_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    estimator: BeatThisEstimator | None,
) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    stem = safe_stem(audio_path)
    file_output = output_root / stem
    file_output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": file_output / f"{stem}__inference.csv",
        "segments": file_output / f"{stem}__segments.csv",
        "frames": file_output / f"{stem}__beat-this__frames.csv",
        "raw": file_output / f"{stem}__beat-this-raw__beats.csv",
        "fused": file_output / f"{stem}__beat-this-fused__beats.csv",
        "normalized": file_output / f"{stem}__beat-this-normalized__beats.csv",
        "grid": file_output / f"{stem}__beat-this__grid.csv",
    }
    ensure_segments_csv(paths["segments"], duration)
    activity_segments = read_segments_csv(paths["segments"], duration)
    ranges = active_ranges(activity_segments, duration)
    blocked_ranges = no_beat_ranges(activity_segments, duration)

    raw_note = "Official 30-second keep-first chunk aggregation"
    if estimator is not None:
        inferred_results, inferred_frames = estimator(y, sr)
        metadata = InferenceMetadata(
            fps=inferred_frames.fps,
            window_seconds=inferred_frames.window_seconds,
            hop_seconds=inferred_frames.hop_seconds,
            overlap_windows=inferred_frames.overlap_windows,
            sample_rate=sr,
            duration_seconds=duration,
            audio_path=str(audio_path.resolve()),
        )
        write_inference_metadata_csv(paths["metadata"], metadata)
        write_frame_predictions_csv(paths["frames"], inferred_frames)
        for result in inferred_results:
            result.beat_times = np.unique(
                result.beat_times[np.isfinite(result.beat_times)]
            )
            midpoints, raw_bpm, smooth_bpm = local_tempo(result.beat_times)
            write_beats_csv(
                paths["raw"] if result.method == "beat-this-raw" else paths["fused"],
                result,
                midpoints,
                raw_bpm,
                smooth_bpm,
                sr,
                ranges,
            )
    else:
        missing_cache = [
            path
            for path in (paths["metadata"], paths["frames"], paths["raw"], paths["fused"])
            if not path.is_file()
        ]
        if missing_cache:
            raise FileNotFoundError(
                "--reuse-inference requires cached files:\n  "
                + "\n  ".join(str(path) for path in missing_cache)
            )

    # From this point onward CSV files are the only source of inference data.
    metadata = read_inference_metadata_csv(paths["metadata"])
    if metadata.sample_rate != sr:
        raise ValueError(
            f"Cached sample rate {metadata.sample_rate} does not match loaded rate {sr}"
        )
    if abs(metadata.duration_seconds - duration) > 0.05:
        raise ValueError(
            "Cached inference duration does not match the current audio file: "
            f"{metadata.duration_seconds:.3f} vs {duration:.3f} seconds"
        )
    if Path(metadata.audio_path).resolve() != audio_path.resolve():
        raise ValueError(
            "Cached inference belongs to a different audio path: "
            f"{metadata.audio_path}"
        )
    frames = read_frame_predictions_csv(paths["frames"], metadata)
    fused_note = (
        "30-second shifted windows with Hann-weighted logit fusion "
        f"and {metadata.hop_seconds:g}-second hop"
    )
    raw_result = read_beats_csv(paths["raw"], "beat-this-raw", raw_note)
    fused_result = read_beats_csv(paths["fused"], "beat-this-fused", fused_note)
    for result, path in (
        (raw_result, paths["raw"]),
        (fused_result, paths["fused"]),
    ):
        midpoints, raw_bpm, smooth_bpm = local_tempo(result.beat_times)
        write_beats_csv(
            path,
            result,
            midpoints,
            raw_bpm,
            smooth_bpm,
            sr,
            ranges,
        )
    raw_result = read_beats_csv(paths["raw"], "beat-this-raw", raw_note)
    fused_result = read_beats_csv(paths["fused"], "beat-this-fused", fused_note)
    normalized_result, grid_decoding = build_normalized_grid(
        fused_result,
        frames,
        ranges,
    )
    normalized_midpoints, normalized_raw, normalized_smooth = local_tempo(
        normalized_result.beat_times
    )
    write_beats_csv(
        paths["normalized"],
        normalized_result,
        normalized_midpoints,
        normalized_raw,
        normalized_smooth,
        sr,
        ranges,
    )
    write_grid_decisions_csv(paths["grid"], grid_decoding)

    # Reload derived CSVs too, so plots/reports cannot diverge from saved data.
    normalized_result = read_beats_csv(
        paths["normalized"],
        "beat-this-normalized",
        normalized_result.note,
    )
    grid_decoding = read_grid_decisions_csv(paths["grid"])
    stored_results = [raw_result, fused_result, normalized_result]
    analyses = [
        analyse_result_from_ranges(result, ranges, duration, args, audio_path)
        for result in stored_results
    ]
    for analysis in analyses:
        result = analysis["result"]
        write_click_track(
            file_output / f"{stem}__{result.method}__clicks.wav", y, sr, result
        )

    write_plot(
        file_output / f"{stem}__tempo.png",
        audio_path.name,
        analyses,
        duration,
        blocked_ranges,
    )
    write_probability_plot(
        file_output / f"{stem}__probabilities.png",
        audio_path.name,
        frames,
        duration,
        blocked_ranges,
    )
    write_grid_plot(
        file_output / f"{stem}__grid.png",
        audio_path.name,
        grid_decoding,
        normalized_result,
        ranges,
        blocked_ranges,
        duration,
    )

    normalized_grid_bpm_parts: list[np.ndarray] = []
    for range_index, (start, end) in enumerate(ranges):
        is_last = range_index == len(ranges) - 1
        mask = (normalized_result.beat_times >= start) & (
            (normalized_result.beat_times <= end)
            if is_last
            else (normalized_result.beat_times < end)
        )
        intervals = np.diff(normalized_result.beat_times[mask])
        if len(intervals):
            normalized_grid_bpm_parts.append(60.0 / intervals)
    normalized_grid_bpm = (
        np.concatenate(normalized_grid_bpm_parts)
        if normalized_grid_bpm_parts
        else np.asarray([], dtype=float)
    )
    scale_usage = []
    for scale in GRID_SCALES:
        mask = grid_decoding.selected_scale == scale
        seconds = (
            float(np.sum(grid_decoding.interval_durations[mask]))
            if len(mask)
            else 0.0
        )
        active_grid_seconds = float(np.sum(grid_decoding.interval_durations))
        scale_usage.append(
            {
                "scale": float(scale),
                "intervals": int(np.sum(mask)),
                "seconds": round(seconds, 3),
                "time_percent": (
                    round(100.0 * seconds / active_grid_seconds, 2)
                    if active_grid_seconds > 0
                    else 0.0
                ),
            }
        )
    report = {
        "audio": str(audio_path.resolve()),
        "duration_seconds": round(duration, 3),
        "activity": {
            "segments_csv": paths["segments"].name,
            "active_seconds": round(sum(end - start for start, end in ranges), 3),
            "no_beat_seconds": round(
                sum(end - start for start, end in blocked_ranges), 3
            ),
            "no_beat_ranges": [
                {"start_seconds": start, "end_seconds": end}
                for start, end in blocked_ranges
            ],
        },
        "frame_inference": {
            "fps": frames.fps,
            "window_seconds": frames.window_seconds,
            "hop_seconds": frames.hop_seconds,
            "overlap_windows": frames.overlap_windows,
            "aggregation": "Hann-weighted logit mean",
        },
        "grid_normalization": {
            "target_bpm_interval": "[120, 240)",
            "candidate_scales": [float(scale) for scale in GRID_SCALES],
            "scale_usage": scale_usage,
            "intervals_outside_target_after_selection": int(
                np.sum(
                    (grid_decoding.normalized_bpm < NORMALIZED_BPM_MIN)
                    | (grid_decoding.normalized_bpm >= NORMALIZED_BPM_MAX)
                )
            ),
            "final_grid_intervals_outside_target": int(
                np.sum(
                    (normalized_grid_bpm < NORMALIZED_BPM_MIN)
                    | (normalized_grid_bpm >= NORMALIZED_BPM_MAX)
                )
            ),
        },
        "interpretation_note": (
            "All plots and reports are regenerated from saved CSV data. "
            "NO_BEAT ranges are excluded from clicks, tempo statistics, and grid decoding. "
            "Highlighted tempo-change regions are candidates, not ground truth. "
            "Exact 2x/0.5x tempo changes are musically ambiguous; raw BPM remains in each beat CSV."
        ),
        "methods": [
            {
                "method": item["result"].method,
                "reported_bpm": item["result"].reported_bpm,
                "dominant_bpm_from_beats": (
                    round(item["main_bpm"], 2) if np.isfinite(item["main_bpm"]) else None
                ),
                "beats": int(len(item["result"].beat_times)),
                "downbeats": int(
                    len(item["result"].downbeat_times)
                    if item["result"].downbeat_times is not None
                    else 0
                ),
                "possible_tempo_change_regions": [asdict(region) for region in item["regions"]],
                "note": item["result"].note,
            }
            for item in analyses
        ],
    }
    with (file_output / f"{stem}__report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return analyses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track beats with Beat This! and inspect changing tempo."
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Audio file(s): WAV/MP3/FLAC/M4A...")
    parser.add_argument("-o", "--output", type=Path, default=Path("bpm_results"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--min-bpm", type=float, default=40.0)
    parser.add_argument("--max-bpm", type=float, default=240.0)
    parser.add_argument("--change-ratio", type=float, default=0.08, help="Relative change threshold (default 8%%)")
    parser.add_argument("--change-bpm", type=float, default=8.0, help="Absolute change threshold")
    parser.add_argument("--min-change-beats", type=int, default=4)
    parser.add_argument("--min-change-seconds", type=float, default=2.0)
    parser.add_argument("--beat-this-device", default="cpu", help="cpu, cuda, cuda:0, mps...")
    parser.add_argument("--beat-this-checkpoint", default="final0")
    parser.add_argument(
        "--beat-this-hop-seconds",
        type=float,
        default=10.0,
        help="Hop between overlapping 30-second inference windows (default: 10)",
    )
    parser.add_argument(
        "--reuse-inference",
        action="store_true",
        help=(
            "Skip Beat This! and rebuild normalized CSVs, clicks, plots, and "
            "reports from cached inference CSVs plus the editable segments CSV"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    missing = [path for path in args.audio if not path.is_file()]
    if missing:
        print("Missing input file(s):", *missing, sep="\n  ", file=sys.stderr)
        return 2
    if not 0 < args.min_bpm < args.max_bpm:
        print("Require 0 < --min-bpm < --max-bpm", file=sys.stderr)
        return 2
    if not 0 < args.beat_this_hop_seconds < 30:
        print("Require 0 < --beat-this-hop-seconds < 30", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    beat_this: BeatThisEstimator | None = None
    if not args.reuse_inference:
        try:
            beat_this = BeatThisEstimator(
                args.beat_this_device,
                args.beat_this_checkpoint,
                hop_seconds=args.beat_this_hop_seconds,
            )
        except Exception as exc:
            print(f"Unable to initialize Beat This!: {exc}", file=sys.stderr)
            return 1

    all_analyses: list[dict] = []
    for audio_path in args.audio:
        print(f"Analysing: {audio_path}")
        try:
            all_analyses.extend(analyse_file(audio_path, args.output, args, beat_this))
        except Exception as exc:
            print(f"Processing failed for {audio_path.name}: {exc}", file=sys.stderr)
            return 1

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audio",
                "method",
                "duration_seconds",
                "active_seconds",
                "no_beat_seconds",
                "reported_bpm",
                "dominant_bpm_from_beats",
                "detected_beats",
                "possible_change_regions",
            ]
        )
        for item in all_analyses:
            result = item["result"]
            writer.writerow(
                [
                    str(item["audio"]),
                    result.method,
                    f"{item['duration']:.3f}",
                    f"{item['active_duration']:.3f}",
                    f"{item['no_beat_duration']:.3f}",
                    "" if result.reported_bpm is None else f"{result.reported_bpm:.3f}",
                    "" if not np.isfinite(item["main_bpm"]) else f"{item['main_bpm']:.3f}",
                    len(result.beat_times),
                    len(item["regions"]),
                ]
            )
    print(f"Done. Open: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
