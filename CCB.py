#!/usr/bin/env python3
"""Generate the final Context Calibrated Beats grid for offline audio.

Each song produces one beat CSV, an optional click track, one overview image,
an editable NO_BEAT segment map, and a concise JSON report. Beat This! inference
is kept in a hidden cache so user edits can be rebuilt without rerunning the
model. The validated phase-aware grid is the only public beat result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib import font_manager
from platformdirs import user_cache_dir
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

DEFAULT_MUSIC_GAIN = 0.1
DEFAULT_CLICK_GAIN = 0.9
CCB_VERSION = "1.0.1"
CACHE_SCHEMA_VERSION = 2


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
class PhaseGridEvent:
    grid_beat_index: int
    activity_segment_id: int
    beat_time_seconds: float
    selected_scale: float
    target_period_seconds: float
    predicted_time_seconds: float
    phase_residual_seconds: float
    beat_probability: float
    downbeat_probability: float
    event_source: str
    transition_type: str
    path_cost: float


@dataclass
class GridTransition:
    transition_id: int
    activity_segment_id: int
    start_seconds: float
    end_seconds: float
    previous_scale: float
    next_scale: float
    previous_period_seconds: float
    next_period_seconds: float
    phase_adjustment_seconds: float
    transition_type: str
    diagnostic_note: str


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
class ReliabilitySegment:
    segment_id: int
    activity_segment_id: int
    start_seconds: float
    end_seconds: float
    classification: str
    reliability_score: float
    reference_bpm: float | None
    local_bpm: float | None
    acoustic_support: float
    scale_switches: int
    phase_repairs: int
    reason: str


@dataclass
class InferenceMetadata:
    fps: float
    window_seconds: float
    hop_seconds: float
    overlap_windows: int
    sample_rate: int
    duration_seconds: float
    audio_path: str
    cache_schema_version: int = 1
    audio_size_bytes: int | None = None
    audio_mtime_ns: int | None = None
    checkpoint: str | None = None
    beat_this_version: str | None = None
    ccb_version: str | None = None


@dataclass
class IntervalDiagnostic:
    activity_segment_id: int
    interval_index: int
    left_beat_index: int
    right_beat_index: int
    start_seconds: float
    end_seconds: float
    midpoint_seconds: float
    observed_interval_seconds: float
    reference_interval_seconds: float
    relative_deviation: float
    acoustic_confidence: float
    timing_confidence: float
    local_stability: float
    classification: str


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
    ) -> tuple[BeatResult, FramePredictions]:
        spect = self.frame_model.signal2spect(y, sr)
        fused_beat_logits, fused_downbeat_logits, overlap_windows = (
            self._overlap_fused_logits(spect)
        )

        fused_beats, fused_downbeats = self.postprocessor(
            fused_beat_logits, fused_downbeat_logits
        )
        result = BeatResult(
            method="beat-this-fused",
            beat_times=np.asarray(fused_beats, dtype=float),
            downbeat_times=np.asarray(fused_downbeats, dtype=float),
            note=(
                "30-second shifted windows with Hann-weighted logit fusion "
                f"and {self.hop_frames / self.FPS:g}-second hop"
            ),
        )
        frames = FramePredictions(
            fps=self.FPS,
            fused_beat_logits=self._numpy(fused_beat_logits),
            fused_downbeat_logits=self._numpy(fused_downbeat_logits),
            window_seconds=self.WINDOW_FRAMES / self.FPS,
            hop_seconds=self.hop_frames / self.FPS,
            overlap_windows=overlap_windows,
        )
        return result, frames


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


def _probability_near(
    time_seconds: float,
    logits: np.ndarray,
    fps: float,
    radius_seconds: float = 0.08,
) -> float:
    if not len(logits):
        return 0.0
    centre = int(round(time_seconds * fps))
    radius = max(0, int(round(radius_seconds * fps)))
    start = max(0, centre - radius)
    end = min(len(logits), centre + radius + 1)
    if start >= end:
        return 0.0
    return float(np.max(expit(logits[start:end])))


def _robust_stability(values: np.ndarray) -> float:
    """Return a 0..1 score; one means a locally constant beat period."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 0.0
    median = float(np.median(values))
    if median <= 1e-9:
        return 0.0
    relative_mad = 1.4826 * float(np.median(np.abs(values - median))) / median
    return float(np.clip(1.0 - relative_mad / 0.10, 0.0, 1.0))


def diagnose_fused_beats(
    fused_result: BeatResult,
    frames: FramePredictions,
    ranges: list[tuple[float, float]],
) -> list[IntervalDiagnostic]:
    """Describe possible beat errors without changing any beat timestamps."""
    all_times = np.asarray(fused_result.beat_times, dtype=float)
    diagnostics: list[IntervalDiagnostic] = []

    for segment_id, (range_start, range_end) in enumerate(ranges):
        is_last = segment_id == len(ranges) - 1
        mask = (all_times >= range_start) & (
            (all_times <= range_end) if is_last else (all_times < range_end)
        )
        global_indices = np.flatnonzero(mask)
        times = all_times[mask]
        if len(times) < 2:
            continue
        intervals = np.diff(times)
        references = np.empty_like(intervals)
        stabilities = np.empty_like(intervals)
        for index in range(len(intervals)):
            left = max(0, index - 4)
            right = min(len(intervals), index + 5)
            neighbours = np.delete(intervals[left:right], index - left)
            if not len(neighbours):
                neighbours = intervals[left:right]
            references[index] = float(np.median(neighbours))
            stabilities[index] = _robust_stability(neighbours)

        deviations = np.divide(
            np.abs(intervals - references),
            references,
            out=np.zeros_like(intervals),
            where=references > 1e-9,
        )

        # Protect smooth, persistent period motion. It is evidence of rubato or a
        # real tempo transition, not an isolated beat error.
        protected = np.zeros(len(intervals), dtype=bool)
        if len(intervals) >= 4:
            log_period = np.log(np.maximum(intervals, 1e-9))
            for start in range(len(intervals) - 3):
                window = log_period[start : start + 4]
                deltas = np.diff(window)
                same_direction = bool(
                    np.all(deltas >= -0.015) or np.all(deltas <= 0.015)
                )
                gradual_steps = bool(np.all(np.abs(deltas) <= 0.16))
                total_motion = abs(float(np.exp(window[-1] - window[0]) - 1.0))
                if same_direction and gradual_steps and total_motion >= 0.12:
                    protected[start : start + 4] = True

        for index, (interval, reference, deviation) in enumerate(
            zip(intervals, references, deviations)
        ):
            midpoint = float((times[index] + times[index + 1]) / 2.0)
            acoustic = 0.5 * (
                _probability_near(
                    float(times[index]), frames.fused_beat_logits, frames.fps, 0.02
                )
                + _probability_near(
                    float(times[index + 1]),
                    frames.fused_beat_logits,
                    frames.fps,
                    0.02,
                )
            )
            timing = float(np.exp(-max(0.0, float(deviation) - 0.10) / 0.15))
            if protected[index]:
                classification = "protected_tempo_motion"
            elif deviation <= 0.10:
                classification = "normal"
            elif deviation <= 0.15:
                classification = "observe"
            elif deviation <= 0.25:
                classification = "structural_evidence_required"
            else:
                classification = "strong_outlier"
            diagnostics.append(
                IntervalDiagnostic(
                    activity_segment_id=segment_id,
                    interval_index=index + 1,
                    left_beat_index=int(global_indices[index]) + 1,
                    right_beat_index=int(global_indices[index + 1]) + 1,
                    start_seconds=float(times[index]),
                    end_seconds=float(times[index + 1]),
                    midpoint_seconds=midpoint,
                    observed_interval_seconds=float(interval),
                    reference_interval_seconds=float(reference),
                    relative_deviation=float(deviation),
                    acoustic_confidence=float(acoustic),
                    timing_confidence=timing,
                    local_stability=float(stabilities[index]),
                    classification=classification,
                )
            )

    return diagnostics


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


def _collapse_repeated_close_doublets(
    segment_times: np.ndarray,
    probabilities,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """Collapse sustained 80-ms-style duplicate pairs before grid decoding.

    A single close pair can be a legitimate transient or an ornament, so it is
    never enough to alter the source used by the grid. This gate only accepts
    at least four consecutive pairs whose pair-to-pair spacing is a stable
    0.70--0.90 seconds. That is the characteristic half-time/double-peak
    topology seen in Aria, while ordinary dense passages and isolated doublets
    remain untouched. Raw/fused inference CSVs are not modified.
    """
    times = np.asarray(segment_times, dtype=float)
    if len(times) < 8:
        return times, []

    pairs: list[tuple[int, int]] = []
    index = 0
    while index + 1 < len(times):
        gap = float(times[index + 1] - times[index])
        left_support = max(probabilities(float(times[index])))
        right_support = max(probabilities(float(times[index + 1])))
        if 0.04 <= gap <= 0.12 and max(left_support, right_support) >= 0.45:
            pairs.append((index, index + 1))
            index += 2
        else:
            index += 1

    runs: list[list[tuple[int, int]]] = []
    for pair in pairs:
        if not runs:
            runs.append([pair])
            continue
        previous = runs[-1][-1]
        start_spacing = float(times[pair[0]] - times[previous[0]])
        consecutive = pair[0] == previous[1] + 1
        if consecutive and 0.70 <= start_spacing <= 0.90:
            runs[-1].append(pair)
        else:
            runs.append([pair])

    drop_indices: set[int] = set()
    collapsed_runs: list[tuple[float, float, int]] = []
    for run in runs:
        if len(run) < 4:
            continue
        starts = np.asarray([times[left] for left, _ in run], dtype=float)
        spacings = np.diff(starts)
        median_spacing = float(np.median(spacings))
        if (
            median_spacing <= 0
            or float(np.max(np.abs(spacings - median_spacing)))
            / median_spacing
            > 0.06
        ):
            continue

        left_scores: list[float] = []
        right_scores: list[float] = []
        for left, right in run:
            left_beat, left_downbeat = probabilities(float(times[left]))
            right_beat, right_downbeat = probabilities(float(times[right]))
            left_scores.append(left_beat + 0.35 * left_downbeat)
            right_scores.append(right_beat + 0.35 * right_downbeat)
        if float(np.mean(np.maximum(left_scores, right_scores))) < 0.55:
            continue

        keep_right = float(np.mean(right_scores)) > float(np.mean(left_scores))
        for left, right in run:
            drop_indices.add(left if keep_right else right)
        collapsed_runs.append(
            (
                float(times[run[0][0]]),
                float(times[run[-1][1]]),
                len(run),
            )
        )

    if not drop_indices:
        return times, []
    keep_mask = np.ones(len(times), dtype=bool)
    keep_mask[list(drop_indices)] = False
    return times[keep_mask], collapsed_runs


def _future_aligned_segment_start(
    segment_times: np.ndarray,
    period_at,
    probabilities,
) -> tuple[float, float, str] | None:
    """Use a stable near-future lattice to disambiguate a segment's first beat.

    This is intentionally a start-phase selector, not a general backtracker:
    it may move the first grid event by less than one normalized period, but it
    never inserts, removes, or rewrites the Beat This! source detections.
    """
    if len(segment_times) < 5:
        return None
    original_start = float(segment_times[0])
    horizon_end = min(float(segment_times[-1]), original_start + 6.0)
    anchors = np.asarray(
        [
            float(time_seconds)
            for time_seconds in segment_times
            if time_seconds <= horizon_end + 1e-9
            and max(probabilities(float(time_seconds))) >= 0.45
        ],
        dtype=float,
    )
    if len(anchors) < 4:
        return None

    runs: list[np.ndarray] = []
    run_start = 0
    for index, interval in enumerate(np.diff(anchors)):
        left = float(anchors[index])
        right = float(anchors[index + 1])
        period = float(np.clip(0.5 * (period_at(left) + period_at(right)), 1e-6, None))
        steps = max(1, int(round(float(interval) / period)))
        residual = abs(float(interval) - steps * period)
        compatible = steps <= 4 and residual <= max(0.045, 0.14 * period)
        if not compatible:
            runs.append(anchors[run_start : index + 1])
            run_start = index + 1
    runs.append(anchors[run_start:])

    candidates: list[tuple[float, int, float, float, float, str]] = []
    for run in runs:
        if len(run) < 4 or run[-1] - run[0] < 1.8:
            continue
        local_periods = np.asarray([period_at(float(item)) for item in run], dtype=float)
        reference_period = float(np.median(local_periods))
        if reference_period <= 0:
            continue
        if float(np.max(np.abs(local_periods - reference_period))) > 0.10 * reference_period:
            continue
        steps = np.maximum(1, np.rint(np.diff(run) / reference_period).astype(int))
        positions = np.r_[0, np.cumsum(steps)].astype(float)
        fitted_period, fitted_phase = np.polyfit(positions, run, 1)
        fitted_period = float(fitted_period)
        fitted_phase = float(fitted_phase)
        if not 0.90 * reference_period <= fitted_period <= 1.10 * reference_period:
            continue
        fitted = fitted_phase + positions * fitted_period
        residuals = np.abs(run - fitted)
        if float(np.max(residuals)) > max(0.035, 0.11 * fitted_period):
            continue
        candidate_start = fitted_phase
        while candidate_start - fitted_period >= original_start - 1e-9:
            candidate_start -= fitted_period
        while candidate_start < original_start - 1e-9:
            candidate_start += fitted_period
        shift = candidate_start - original_start
        if shift <= max(0.055, 0.16 * fitted_period):
            continue

        def lattice_residual(time_seconds: float, phase: float) -> float:
            lattice_step = round((time_seconds - phase) / fitted_period)
            return abs(time_seconds - (phase + lattice_step * fitted_period))

        old_error = float(
            np.median(
                [lattice_residual(float(item), original_start) for item in run]
            )
        )
        new_error = float(
            np.median(
                [lattice_residual(float(item), candidate_start) for item in run]
            )
        )
        improvement = old_error - new_error
        if old_error < 0.22 * fitted_period or improvement < 0.15 * fitted_period:
            continue
        support = float(
            np.mean([max(probabilities(float(item))) for item in run])
        )
        if support < 0.65:
            continue
        note = (
            f"Stable future lattice at {run[0]:.3f}-{run[-1]:.3f}s "
            f"moved segment start by {shift:.3f}s; "
            f"median phase error {old_error:.3f}->{new_error:.3f}s"
        )
        candidates.append(
            (new_error, -len(run), float(run[0]), candidate_start, fitted_period, note)
        )
    if not candidates:
        return None
    _, _, _, candidate_start, fitted_period, note = min(candidates)
    return float(candidate_start), float(fitted_period), note


def build_phase_aware_grid(
    repaired_result: BeatResult,
    frames: FramePredictions,
    active_ranges: list[tuple[float, float]],
) -> tuple[BeatResult, list[PhaseGridEvent], list[GridTransition]]:
    """Track one continuous phase path while using only the five grid scales."""
    source_times = np.asarray(repaired_result.beat_times, dtype=float)
    source_downbeats = np.asarray(
        repaired_result.downbeat_times
        if repaired_result.downbeat_times is not None
        else [],
        dtype=float,
    )
    beat_probability = expit(frames.fused_beat_logits)
    downbeat_probability = expit(frames.fused_downbeat_logits)
    minimum_period = 60.0 / NORMALIZED_BPM_MAX + 1e-4
    maximum_period = 60.0 / NORMALIZED_BPM_MIN
    events: list[PhaseGridEvent] = []
    transitions: list[GridTransition] = []
    output_downbeats: list[float] = []
    path_cost = 0.0

    def probabilities(time_seconds: float) -> tuple[float, float]:
        index = _frame_index(time_seconds, frames)
        return float(beat_probability[index]), float(downbeat_probability[index])

    for segment_id, (range_start, range_end) in enumerate(active_ranges):
        is_last = segment_id == len(active_ranges) - 1
        mask = (source_times >= range_start) & (
            (source_times <= range_end) if is_last else (source_times < range_end)
        )
        segment_times = source_times[mask]
        if len(segment_times) < 2:
            continue
        segment_times, collapsed_doublet_runs = _collapse_repeated_close_doublets(
            segment_times,
            probabilities,
        )
        if len(segment_times) < 2:
            continue
        midpoints, _, base_bpm = local_tempo(segment_times)
        states, _ = decode_grid_scales(base_bpm)
        scales = GRID_SCALES[states]
        target_periods = np.clip(
            60.0 / np.maximum(base_bpm * scales, 1e-9),
            minimum_period,
            maximum_period,
        )

        def interval_index(time_seconds: float) -> int:
            return int(
                np.clip(
                    np.searchsorted(midpoints, time_seconds, side="right") - 1,
                    0,
                    len(midpoints) - 1,
                )
            )

        def scale_at(time_seconds: float) -> float:
            return float(scales[interval_index(time_seconds)])

        def period_at(time_seconds: float) -> float:
            if len(midpoints) == 1:
                return float(target_periods[0])
            return float(
                np.interp(
                    time_seconds,
                    midpoints,
                    target_periods,
                    left=target_periods[0],
                    right=target_periods[-1],
                )
            )

        for run_start, run_end, pair_count in collapsed_doublet_runs:
            transitions.append(
                GridTransition(
                    transition_id=len(transitions) + 1,
                    activity_segment_id=segment_id,
                    start_seconds=run_start,
                    end_seconds=run_end,
                    previous_scale=scale_at(run_start),
                    next_scale=scale_at(run_end),
                    previous_period_seconds=period_at(run_start),
                    next_period_seconds=period_at(run_end),
                    phase_adjustment_seconds=0.0,
                    transition_type="source_doublet_disambiguation",
                    diagnostic_note=(
                        f"Collapsed {pair_count} consecutive close source pairs "
                        "before phase-grid decoding"
                    ),
                )
            )

        original_start = float(segment_times[0])
        future_start = _future_aligned_segment_start(
            segment_times,
            period_at,
            probabilities,
        )
        current = future_start[0] if future_start is not None else original_start
        current_scale = scale_at(current)
        current_period = (
            future_start[1] if future_start is not None else period_at(current)
        )
        beat_prob, downbeat_prob = probabilities(current)
        events.append(
            PhaseGridEvent(
                grid_beat_index=len(events) + 1,
                activity_segment_id=segment_id,
                beat_time_seconds=current,
                selected_scale=current_scale,
                target_period_seconds=current_period,
                predicted_time_seconds=current,
                phase_residual_seconds=0.0,
                beat_probability=beat_prob,
                downbeat_probability=downbeat_prob,
                event_source=(
                    "future_phase_start"
                    if future_start is not None
                    else "acoustic_anchor"
                ),
                transition_type=(
                    "segment_start_future_phase"
                    if future_start is not None
                    else "segment_start"
                ),
                path_cost=path_cost,
            )
        )
        if future_start is not None:
            transitions.append(
                GridTransition(
                    transition_id=len(transitions) + 1,
                    activity_segment_id=segment_id,
                    start_seconds=original_start,
                    end_seconds=current,
                    previous_scale=scale_at(original_start),
                    next_scale=current_scale,
                    previous_period_seconds=period_at(original_start),
                    next_period_seconds=current_period,
                    phase_adjustment_seconds=current - original_start,
                    transition_type="future_phase_start",
                    diagnostic_note=future_start[2],
                )
            )
        if (
            len(source_downbeats)
            and np.min(np.abs(source_downbeats - current)) <= 0.07
        ) or downbeat_prob >= 0.5:
            output_downbeats.append(current)

        previous_interval: float | None = None
        previous_scale = current_scale
        max_events = int(math.ceil((segment_times[-1] - current) / minimum_period)) + 4
        for _ in range(max_events):
            first_period = period_at(current)
            predicted = current + first_period
            target_period = float(
                np.clip(
                    0.5 * (first_period + period_at(predicted)),
                    minimum_period,
                    maximum_period,
                )
            )
            if previous_interval is not None:
                target_period = float(
                    np.clip(
                        target_period,
                        max(minimum_period, 0.85 * previous_interval),
                        min(maximum_period, 1.15 * previous_interval),
                    )
                )
            predicted = current + target_period
            if predicted > segment_times[-1] + 1e-6:
                tail = float(segment_times[-1] - current)
                if minimum_period <= tail <= maximum_period:
                    predicted = float(segment_times[-1])
                    target_period = tail
                else:
                    break

            acquisition_radius = min(0.18, 0.55 * target_period)
            nearby = segment_times[
                (segment_times >= predicted - acquisition_radius)
                & (segment_times <= predicted + acquisition_radius)
                & (segment_times > current + minimum_period - 1e-6)
            ]
            predicted_beat, _ = probabilities(predicted)
            best_time: float | None = None
            best_score = 0.10 + 0.25 * predicted_beat
            for candidate_time in nearby:
                candidate_beat, candidate_downbeat = probabilities(
                    float(candidate_time)
                )
                phase_distance = abs(float(candidate_time) - predicted) / target_period
                score = (
                    candidate_beat
                    + 0.25 * candidate_downbeat
                    - 1.10 * phase_distance
                )
                if score > best_score + 0.05:
                    best_score = score
                    best_time = float(candidate_time)

            event_source = "theoretical_grid"
            next_time = predicted
            if best_time is not None:
                residual = best_time - predicted
                snap_radius = min(0.08, 0.24 * target_period)
                maximum_adjustment = min(0.05, 0.15 * target_period)
                if abs(residual) <= snap_radius:
                    next_time = best_time
                    event_source = "acoustic_peak"
                else:
                    next_time = predicted + float(
                        np.clip(residual, -maximum_adjustment, maximum_adjustment)
                    )
                    event_source = "phase_bridge"

            interval = next_time - current
            lower = minimum_period
            upper = maximum_period
            if previous_interval is not None:
                lower = max(lower, 0.85 * previous_interval)
                upper = min(upper, 1.15 * previous_interval)
            constrained_interval = float(np.clip(interval, lower, upper))
            if abs(constrained_interval - interval) > 1e-9:
                next_time = current + constrained_interval
                event_source = "phase_bridge"
            if next_time > segment_times[-1] + 1e-6:
                break
            interval = next_time - current
            if interval < minimum_period - 1e-6 or interval > maximum_period + 1e-6:
                raise ValueError("Phase-aware decoder produced an invalid period")

            selected_scale = scale_at(next_time)
            phase_residual = next_time - predicted
            beat_prob, downbeat_prob = probabilities(next_time)
            scale_changed = selected_scale != previous_scale
            transition_type = (
                "scale_switch"
                if scale_changed
                else "phase_bridge"
                if event_source == "phase_bridge"
                else "stable"
            )
            path_cost += (
                abs(interval - target_period) / max(target_period, 1e-9)
                + 0.35 * (1.0 - beat_prob)
                + (0.12 if scale_changed else 0.0)
            )
            events.append(
                PhaseGridEvent(
                    grid_beat_index=len(events) + 1,
                    activity_segment_id=segment_id,
                    beat_time_seconds=float(next_time),
                    selected_scale=selected_scale,
                    target_period_seconds=target_period,
                    predicted_time_seconds=predicted,
                    phase_residual_seconds=phase_residual,
                    beat_probability=beat_prob,
                    downbeat_probability=downbeat_prob,
                    event_source=event_source,
                    transition_type=transition_type,
                    path_cost=path_cost,
                )
            )
            if (
                len(source_downbeats)
                and np.min(np.abs(source_downbeats - next_time)) <= 0.07
            ) or downbeat_prob >= 0.5:
                output_downbeats.append(float(next_time))
            if scale_changed or event_source == "phase_bridge":
                transitions.append(
                    GridTransition(
                        transition_id=len(transitions) + 1,
                        activity_segment_id=segment_id,
                        start_seconds=current,
                        end_seconds=float(next_time),
                        previous_scale=previous_scale,
                        next_scale=selected_scale,
                        previous_period_seconds=(
                            previous_interval
                            if previous_interval is not None
                            else target_period
                        ),
                        next_period_seconds=interval,
                        phase_adjustment_seconds=phase_residual,
                        transition_type=transition_type,
                        diagnostic_note=(
                            "Phase was adjusted gradually toward acoustic evidence"
                            if event_source == "phase_bridge"
                            else "Metrical scale changed without resetting phase"
                        ),
                    )
                )
            previous_interval = interval
            previous_scale = selected_scale
            current = float(next_time)
            if segment_times[-1] - current < minimum_period - 1e-6:
                break

    output_times = np.asarray([item.beat_time_seconds for item in events], dtype=float)
    for segment_id in range(len(active_ranges)):
        segment_event_times = np.asarray(
            [
                item.beat_time_seconds
                for item in events
                if item.activity_segment_id == segment_id
            ],
            dtype=float,
        )
        if len(segment_event_times) < 2:
            continue
        intervals = np.diff(segment_event_times)
        if np.any(intervals < minimum_period - 1e-6) or np.any(
            intervals > maximum_period + 1e-6
        ):
            raise ValueError("Phase-aware final grid failed canonical range validation")
    result = BeatResult(
        method="beat-this-phase-aware",
        beat_times=output_times,
        downbeat_times=np.unique(np.asarray(output_downbeats, dtype=float)),
        note=(
            "CCB phase-continuous [120, 240) grid using only "
            "0.25x/0.5x/1x/2x/4x metrical states"
        ),
    )
    return result, events, transitions


def reconcile_anchor_beat_counts(
    events: list[PhaseGridEvent],
    transitions: list[GridTransition],
    frames: FramePredictions,
) -> tuple[list[PhaseGridEvent], list[GridTransition]]:
    """Repair a one-beat count error between two short, stable anchor windows."""
    reconciled = [PhaseGridEvent(**asdict(item)) for item in events]
    reconciled_transitions = [
        GridTransition(**asdict(item)) for item in transitions
    ]
    beat_probability = expit(frames.fused_beat_logits)
    downbeat_probability = expit(frames.fused_downbeat_logits)
    minimum_period = 60.0 / NORMALIZED_BPM_MAX + 1e-4
    maximum_period = 60.0 / NORMALIZED_BPM_MIN
    suspect_types = {"scale_switch", "phase_bridge", "bidirectional_bridge"}

    def probabilities(time_seconds: float) -> tuple[float, float]:
        index = _frame_index(time_seconds, frames)
        return float(beat_probability[index]), float(downbeat_probability[index])

    def downbeat_anchor_evidence(
        item: PhaseGridEvent,
    ) -> tuple[float, float, float]:
        """Return the strongest nearby joint beat/downbeat evidence.

        Phase bridges can land on the shoulder of a broad downbeat peak. Use
        the nearby frame maximum to qualify and time a bar anchor, while still
        keeping the phase-grid event itself fixed unless count reconciliation
        is independently proven safe.
        """
        phase_adjusted = (
            item.event_source in {"phase_bridge", "bidirectional_refined"}
            or item.transition_type
            in {"phase_bridge", "scale_switch", "bidirectional_bridge"}
        )
        if (
            not phase_adjusted
            and item.downbeat_probability >= 0.75
            and item.beat_probability >= 0.45
        ):
            return (
                item.beat_time_seconds,
                item.beat_probability,
                item.downbeat_probability,
            )
        if item.downbeat_probability < 0.45:
            return (
                item.beat_time_seconds,
                item.beat_probability,
                item.downbeat_probability,
            )

        radius_frames = max(1, int(math.ceil(0.12 * frames.fps)))
        center = _frame_index(item.beat_time_seconds, frames)
        start = max(0, center - radius_frames)
        end = min(len(beat_probability), center + radius_frames + 1)
        local_beat = beat_probability[start:end]
        local_downbeat = downbeat_probability[start:end]
        eligible = np.flatnonzero(local_beat >= 0.45)
        if len(eligible):
            local_index = int(eligible[np.argmax(local_downbeat[eligible])])
            frame_index = start + local_index
            nearby_time = float(frame_index / frames.fps)
            nearby_beat = float(beat_probability[frame_index])
            nearby_downbeat = float(downbeat_probability[frame_index])
            if nearby_downbeat > item.downbeat_probability:
                return nearby_time, nearby_beat, nearby_downbeat
        return (
            item.beat_time_seconds,
            item.beat_probability,
            item.downbeat_probability,
        )

    def unstable_event(item: PhaseGridEvent) -> bool:
        return (
            item.event_source in {"phase_bridge", "bidirectional_refined"}
            or item.transition_type
            in {"phase_bridge", "scale_switch", "bidirectional_bridge"}
        )

    candidates: list[
        tuple[int, int, int, int, float, float, int, int, str]
    ] = []
    for segment_id in sorted({item.activity_segment_id for item in reconciled}):
        segment_indices = [
            index
            for index, item in enumerate(reconciled)
            if item.activity_segment_id == segment_id
        ]
        if len(segment_indices) < 12:
            continue
        local_events = [reconciled[index] for index in segment_indices]
        local_transitions = sorted(
            [
                item
                for item in reconciled_transitions
                if item.activity_segment_id == segment_id
                and item.transition_type in suspect_types
            ],
            key=lambda item: (item.start_seconds, item.end_seconds),
        )
        clusters: list[list[GridTransition]] = []
        for item in local_transitions:
            current_end = (
                max(value.end_seconds for value in clusters[-1])
                if clusters
                else -math.inf
            )
            current_start = (
                min(value.start_seconds for value in clusters[-1])
                if clusters
                else math.inf
            )
            if (
                clusters
                and item.start_seconds <= current_end + 1.0
                and max(current_end, item.end_seconds)
                - min(current_start, item.start_seconds)
                <= 6.0
            ):
                clusters[-1].append(item)
            else:
                clusters.append([item])

        local_times = np.asarray(
            [item.beat_time_seconds for item in local_events], dtype=float
        )
        for cluster in clusters:
            scale_switches = sum(
                item.transition_type == "scale_switch" for item in cluster
            )
            bridge_steps = sum(
                item.transition_type in {"phase_bridge", "bidirectional_bridge"}
                for item in cluster
            )
            if len(cluster) < 3 or (scale_switches < 2 and bridge_steps < 3):
                continue
            cluster_start = min(item.start_seconds for item in cluster)
            cluster_end = max(item.end_seconds for item in cluster)
            if not 1.5 <= cluster_end - cluster_start <= 6.0:
                continue
            left = int(np.searchsorted(local_times, cluster_start, side="right") - 1)
            nearest_left = int(np.argmin(np.abs(local_times - cluster_start)))
            if abs(local_times[nearest_left] - cluster_start) <= 0.06:
                left = nearest_left
            right = int(np.searchsorted(local_times, cluster_end, side="left"))
            left = max(0, left)
            right = min(len(local_events) - 1, right)
            while right < len(local_events) - 1 and unstable_event(local_events[right]):
                right += 1
            if left < 6 or right + 6 >= len(local_events) or right - left < 4:
                continue
            left_window = local_events[left - 6 : left + 1]
            right_window = local_events[right : right + 7]
            if any(unstable_event(item) for item in left_window + right_window):
                continue
            left_times = np.asarray(
                [item.beat_time_seconds for item in left_window], dtype=float
            )
            right_times = np.asarray(
                [item.beat_time_seconds for item in right_window], dtype=float
            )
            left_intervals = np.diff(left_times)
            right_intervals = np.diff(right_times)
            left_period = float(np.median(left_intervals))
            right_period = float(np.median(right_intervals))
            if (
                _robust_stability(left_intervals) < 0.72
                or _robust_stability(right_intervals) < 0.72
                or abs(left_period - right_period)
                / max(left_period, right_period)
                > 0.06
            ):
                continue
            left_support = sum(
                max(item.beat_probability, item.downbeat_probability) >= 0.45
                for item in left_window
            )
            right_support = sum(
                max(item.beat_probability, item.downbeat_probability) >= 0.45
                for item in right_window
            )
            # Half-time source detections may support only one side directly;
            # the other side can still be a strong timing anchor when its
            # intervals are highly stable. Require one acoustically anchored
            # side and sufficient evidence across both windows.
            if max(left_support, right_support) < 2 or left_support + right_support < 3:
                continue
            left_time = local_events[left].beat_time_seconds
            right_time = local_events[right].beat_time_seconds
            span = right_time - left_time
            if not 1.5 <= span <= 6.0:
                continue
            reference_period = 0.5 * (left_period + right_period)
            expected_intervals = int(round(span / reference_period))
            current_intervals = right - left
            if expected_intervals < 4 or abs(current_intervals - expected_intervals) != 1:
                continue
            corrected_period = span / expected_intervals
            if not minimum_period <= corrected_period <= maximum_period:
                continue
            corrected_error = max(
                abs(corrected_period - left_period) / left_period,
                abs(corrected_period - right_period) / right_period,
            )
            current_period = span / current_intervals
            current_error = max(
                abs(current_period - left_period) / left_period,
                abs(current_period - right_period) / right_period,
            )
            if corrected_error > 0.06 or current_error - corrected_error < 0.025:
                continue
            candidates.append(
                (
                    segment_indices[left],
                    segment_indices[right],
                    segment_id,
                    expected_intervals,
                    left_period,
                    right_period,
                    current_intervals,
                    len(cluster),
                    "stable_windows",
                )
            )

        # A locally unstable passage may not provide the six clean events
        # required above, while its bar structure is still unambiguous. Four
        # strong consecutive downbeats provide three adjacent bar spans; only
        # reconcile the middle bar when the outer bars agree on both duration
        # and beat count and the middle differs by exactly one beat.
        downbeat_evidence = [downbeat_anchor_evidence(item) for item in local_events]
        downbeat_positions = [
            index
            for index, (_, beat_support, downbeat_support) in enumerate(
                downbeat_evidence
            )
            if downbeat_support >= 0.75 and beat_support >= 0.45
        ]
        for anchor_offset in range(len(downbeat_positions) - 3):
            anchor_positions = downbeat_positions[anchor_offset : anchor_offset + 4]
            anchor_times = np.asarray(
                [downbeat_evidence[index][0] for index in anchor_positions],
                dtype=float,
            )
            bar_spans = np.diff(anchor_times)
            median_span = float(np.median(bar_spans))
            if (
                median_span <= 0
                or float(np.max(np.abs(bar_spans - median_span)))
                / median_span
                > 0.06
            ):
                continue
            bar_counts = np.diff(np.asarray(anchor_positions, dtype=int))
            expected_intervals = int(bar_counts[0])
            current_intervals = int(bar_counts[1])
            if (
                expected_intervals != int(bar_counts[2])
                or not 2 <= expected_intervals <= 8
                or abs(current_intervals - expected_intervals) != 1
            ):
                continue
            left_period = float(bar_spans[0] / expected_intervals)
            right_period = float(bar_spans[2] / expected_intervals)
            corrected_period = float(bar_spans[1] / expected_intervals)
            current_period = float(bar_spans[1] / current_intervals)
            if (
                not minimum_period <= corrected_period <= maximum_period
                or abs(left_period - right_period)
                / max(left_period, right_period)
                > 0.06
                or max(
                    abs(corrected_period - left_period) / left_period,
                    abs(corrected_period - right_period) / right_period,
                )
                > 0.06
            ):
                continue
            corrected_error = max(
                abs(corrected_period - left_period) / left_period,
                abs(corrected_period - right_period) / right_period,
            )
            current_error = max(
                abs(current_period - left_period) / left_period,
                abs(current_period - right_period) / right_period,
            )
            if current_error - corrected_error < 0.025:
                continue
            left = int(anchor_positions[1])
            right = int(anchor_positions[2])
            middle_events = local_events[left + 1 : right]
            middle_transitions = [
                item
                for item in local_transitions
                if item.end_seconds > anchor_times[1]
                and item.start_seconds < anchor_times[2]
            ]
            unstable_count = sum(unstable_event(item) for item in middle_events)
            low_support_count = sum(
                max(item.beat_probability, item.downbeat_probability) < 0.45
                for item in middle_events
            )
            if (
                unstable_count < 2
                and low_support_count < 2
                and len(middle_transitions) < 2
            ):
                continue
            candidates.append(
                (
                    segment_indices[left],
                    segment_indices[right],
                    segment_id,
                    expected_intervals,
                    left_period,
                    right_period,
                    current_intervals,
                    unstable_count + low_support_count + len(middle_transitions),
                    "downbeat_bars",
                )
            )

    occupied: list[tuple[int, int]] = []
    for (
        left,
        right,
        segment_id,
        expected_intervals,
        left_period,
        right_period,
        current_intervals,
        evidence_count,
        evidence_source,
    ) in sorted(candidates, reverse=True):
        if any(not (right <= used_left or left >= used_right) for used_left, used_right in occupied):
            continue
        left_event = reconciled[left]
        right_event = reconciled[right]
        left_time = left_event.beat_time_seconds
        right_time = right_event.beat_time_seconds
        corrected_period = (right_time - left_time) / expected_intervals
        old_middle = reconciled[left + 1 : right]
        right_scale = right_event.selected_scale
        scale_switch_time = right_time
        if left_event.selected_scale != right_scale:
            for index, item in enumerate(old_middle):
                if item.selected_scale != right_scale:
                    continue
                suffix = old_middle[index:]
                if all(value.selected_scale == right_scale for value in suffix):
                    scale_switch_time = item.beat_time_seconds
                    break
        replacement: list[PhaseGridEvent] = []
        projected_times: list[float] = []
        for offset in range(1, expected_intervals):
            projected_time = left_time + offset * corrected_period
            projected_times.append(float(projected_time))
            template = min(
                old_middle or [left_event, right_event],
                key=lambda item: abs(item.beat_time_seconds - projected_time),
            )
            beat_prob, downbeat_prob = probabilities(projected_time)
            replacement.append(
                PhaseGridEvent(
                    grid_beat_index=template.grid_beat_index,
                    activity_segment_id=segment_id,
                    beat_time_seconds=float(projected_time),
                    selected_scale=(
                        right_scale
                        if projected_time >= scale_switch_time
                        else left_event.selected_scale
                    ),
                    target_period_seconds=float(corrected_period),
                    predicted_time_seconds=float(projected_time),
                    phase_residual_seconds=0.0,
                    beat_probability=beat_prob,
                    downbeat_probability=downbeat_prob,
                    event_source="anchor_count_reconciled",
                    transition_type="anchor_count_reconciled",
                    path_cost=template.path_cost,
                )
            )
        old_times = [item.beat_time_seconds for item in old_middle]
        nearest_shifts = [
            min(abs(old_time - projected_time) for projected_time in projected_times)
            for old_time in old_times
        ] + [
            min(abs(projected_time - old_time) for old_time in old_times)
            for projected_time in projected_times
        ]
        maximum_adjustment = max(nearest_shifts) if nearest_shifts else 0.0
        reconciled[left + 1 : right] = replacement
        reconciled_transitions = [
            item
            for item in reconciled_transitions
            if not (
                item.activity_segment_id == segment_id
                and item.transition_type in suspect_types
                and left_time
                <= 0.5 * (item.start_seconds + item.end_seconds)
                <= right_time
            )
        ]
        reconciled_transitions.append(
            GridTransition(
                transition_id=len(reconciled_transitions) + 1,
                activity_segment_id=segment_id,
                start_seconds=left_time,
                end_seconds=right_time,
                previous_scale=left_event.selected_scale,
                next_scale=right_scale,
                previous_period_seconds=left_period,
                next_period_seconds=right_period,
                phase_adjustment_seconds=float(maximum_adjustment),
                transition_type="anchor_count_reconciled",
                diagnostic_note=(
                    "Stable anchor windows reconciled interval count "
                    f"{current_intervals}->{expected_intervals}; "
                    f"period={corrected_period:.6f}s; "
                    f"max_shift={maximum_adjustment:.6f}s; "
                    f"evidence={evidence_count}; source={evidence_source}"
                ),
            )
        )
        occupied.append((left, right))

    for grid_index, item in enumerate(reconciled, start=1):
        item.grid_beat_index = grid_index
    reconciled_transitions.sort(
        key=lambda item: (item.activity_segment_id, item.start_seconds, item.end_seconds)
    )
    for transition_id, item in enumerate(reconciled_transitions, start=1):
        item.transition_id = transition_id
    return reconciled, reconciled_transitions


def refine_phase_grid_bidirectionally(
    phase_result: BeatResult,
    events: list[PhaseGridEvent],
    transitions: list[GridTransition],
    frames: FramePredictions,
) -> tuple[BeatResult, list[PhaseGridEvent], list[GridTransition]]:
    """Refine bridge beats and backtrack from independently stable future phase."""
    refined = [PhaseGridEvent(**asdict(item)) for item in events]
    refined_transitions = [GridTransition(**asdict(item)) for item in transitions]
    minimum_period = 60.0 / NORMALIZED_BPM_MAX + 1e-4
    maximum_period = 60.0 / NORMALIZED_BPM_MIN
    beat_probability = expit(frames.fused_beat_logits)
    downbeat_probability = expit(frames.fused_downbeat_logits)
    original_downbeats = np.asarray(
        phase_result.downbeat_times
        if phase_result.downbeat_times is not None
        else [],
        dtype=float,
    )

    def probabilities(time_seconds: float) -> tuple[float, float]:
        index = _frame_index(time_seconds, frames)
        return float(beat_probability[index]), float(downbeat_probability[index])

    def event_cost(
        local_index: int,
        candidate_time: float,
        original_time: float,
        local_events: list[PhaseGridEvent],
    ) -> float:
        beat_prob, downbeat_prob = probabilities(candidate_time)
        evidence = min(1.0, beat_prob + 0.25 * downbeat_prob)
        shift = (candidate_time - original_time) / 0.08
        return 0.90 * (1.0 - evidence) + 0.12 * shift**2

    def path_cost(
        path_times: list[float], local_events: list[PhaseGridEvent]
    ) -> float:
        total = 0.0
        previous_interval: float | None = None
        for index, time_seconds in enumerate(path_times):
            if 0 < index < len(path_times) - 1:
                total += event_cost(
                    index,
                    time_seconds,
                    local_events[index].beat_time_seconds,
                    local_events,
                )
            if index == 0:
                continue
            interval = time_seconds - path_times[index - 1]
            target = 0.5 * (
                local_events[index - 1].target_period_seconds
                + local_events[index].target_period_seconds
            )
            total += 1.35 * math.log(
                max(interval, 1e-9) / max(target, 1e-9)
            ) ** 2
            if previous_interval is not None:
                total += 3.50 * math.log(
                    max(interval, 1e-9) / max(previous_interval, 1e-9)
                ) ** 2
            previous_interval = interval
        return total

    windows: list[tuple[int, int]] = []
    for segment_id in sorted({item.activity_segment_id for item in refined}):
        segment_indices = [
            index
            for index, item in enumerate(refined)
            if item.activity_segment_id == segment_id
        ]
        unstable_positions = [
            position
            for position, global_index in enumerate(segment_indices)
            if refined[global_index].event_source == "phase_bridge"
            or refined[global_index].transition_type == "scale_switch"
        ]
        if not unstable_positions:
            continue
        runs = np.split(
            np.asarray(unstable_positions, dtype=int),
            np.flatnonzero(np.diff(unstable_positions) > 1) + 1,
        )
        for run in runs:
            if not len(run):
                continue
            local_start = max(0, int(run[0]) - 1)
            local_end = min(len(segment_indices) - 1, int(run[-1]) + 1)
            if local_end - local_start >= 2:
                windows.append(
                    (segment_indices[local_start], segment_indices[local_end])
                )

    # Merge overlapping transition windows so each beat is optimized once.
    merged_windows: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged_windows and start <= merged_windows[-1][1]:
            merged_windows[-1] = (
                merged_windows[-1][0],
                max(merged_windows[-1][1], end),
            )
        else:
            merged_windows.append((start, end))

    for start, end in merged_windows:
        local_events = refined[start : end + 1]
        if len(local_events) < 3:
            continue
        candidate_times: list[list[float]] = []
        for index, item in enumerate(local_events):
            if index in {0, len(local_events) - 1}:
                candidate_times.append([item.beat_time_seconds])
                continue
            offsets = np.arange(-0.08, 0.0801, 1.0 / frames.fps)
            values = np.r_[
                item.beat_time_seconds,
                item.beat_time_seconds + offsets,
            ]
            values = np.unique(np.round(values, 9))
            candidate_times.append([float(value) for value in values])

        # Beam search retains second-order interval continuity while optimizing
        # every bridge beat jointly between two fixed anchors.
        beam: list[tuple[float, list[float], float]] = []
        first = candidate_times[0][0]
        for second in candidate_times[1]:
            interval = second - first
            if not minimum_period <= interval <= maximum_period:
                continue
            target = 0.5 * (
                local_events[0].target_period_seconds
                + local_events[1].target_period_seconds
            )
            cost = 1.35 * math.log(interval / max(target, 1e-9)) ** 2
            if len(local_events) > 2:
                cost += event_cost(
                    1,
                    second,
                    local_events[1].beat_time_seconds,
                    local_events,
                )
            beam.append((cost, [first, second], interval))
        for position in range(2, len(local_events)):
            expanded: list[tuple[float, list[float], float]] = []
            for cost, path, previous_interval in beam:
                for candidate_time in candidate_times[position]:
                    interval = candidate_time - path[-1]
                    if not minimum_period <= interval <= maximum_period:
                        continue
                    ratio = interval / previous_interval
                    if not 0.85 <= ratio <= 1.15:
                        continue
                    target = 0.5 * (
                        local_events[position - 1].target_period_seconds
                        + local_events[position].target_period_seconds
                    )
                    added = 1.35 * math.log(
                        interval / max(target, 1e-9)
                    ) ** 2
                    added += 3.50 * math.log(interval / previous_interval) ** 2
                    if position < len(local_events) - 1:
                        added += event_cost(
                            position,
                            candidate_time,
                            local_events[position].beat_time_seconds,
                            local_events,
                        )
                    expanded.append(
                        (cost + added, path + [candidate_time], interval)
                    )
            beam = sorted(expanded, key=lambda item: item[0])[:500]
            if not beam:
                break
        if not beam:
            continue
        def outside_boundary_cost(path: list[float]) -> float:
            cost = 0.0
            if (
                start > 0
                and refined[start - 1].activity_segment_id
                == refined[start].activity_segment_id
            ):
                outside = path[0] - refined[start - 1].beat_time_seconds
                inside = path[1] - path[0]
                ratio = inside / outside
                if not 0.85 <= ratio <= 1.15:
                    return math.inf
                cost += 3.50 * math.log(ratio) ** 2
            if (
                end + 1 < len(refined)
                and refined[end + 1].activity_segment_id
                == refined[end].activity_segment_id
            ):
                inside = path[-1] - path[-2]
                outside = refined[end + 1].beat_time_seconds - path[-1]
                ratio = outside / inside
                if not 0.85 <= ratio <= 1.15:
                    return math.inf
                cost += 3.50 * math.log(ratio) ** 2
            return cost

        scored_beam = [
            (cost + outside_boundary_cost(path), path, interval)
            for cost, path, interval in beam
        ]
        scored_beam = [item for item in scored_beam if np.isfinite(item[0])]
        if not scored_beam:
            continue
        best_cost, best_path, _ = min(scored_beam, key=lambda item: item[0])
        original_path = [item.beat_time_seconds for item in local_events]
        original_cost = path_cost(original_path, local_events) + outside_boundary_cost(
            original_path
        )
        if original_cost - best_cost < 0.03 * max(original_cost, 1.0):
            continue
        shifts = np.abs(np.asarray(best_path) - np.asarray(original_path))
        if float(np.max(shifts)) > 0.080001:
            continue
        changed = False
        for offset, (item, new_time) in enumerate(zip(local_events, best_path)):
            if offset in {0, len(local_events) - 1}:
                continue
            if abs(item.beat_time_seconds - new_time) < 1e-6:
                continue
            changed = True
            global_item = refined[start + offset]
            global_item.beat_time_seconds = float(new_time)
            global_item.phase_residual_seconds = (
                float(new_time) - global_item.predicted_time_seconds
            )
            beat_prob, downbeat_prob = probabilities(float(new_time))
            global_item.beat_probability = beat_prob
            global_item.downbeat_probability = downbeat_prob
            global_item.event_source = "bidirectional_refined"
            global_item.transition_type = "bidirectional_bridge"
        if changed:
            refined_transitions.append(
                GridTransition(
                    transition_id=len(refined_transitions) + 1,
                    activity_segment_id=local_events[0].activity_segment_id,
                    start_seconds=best_path[0],
                    end_seconds=best_path[-1],
                    previous_scale=local_events[0].selected_scale,
                    next_scale=local_events[-1].selected_scale,
                    previous_period_seconds=best_path[1] - best_path[0],
                    next_period_seconds=best_path[-1] - best_path[-2],
                    phase_adjustment_seconds=float(np.max(shifts)),
                    transition_type="bidirectional_bridge",
                    diagnostic_note=(
                        f"Joint anchor refinement reduced local path cost from "
                        f"{original_cost:.4f} to {best_cost:.4f}"
                    ),
                )
            )

    # A fixed-anchor bridge cannot repair a longer region that has locked to a
    # coherent but wrong phase, especially when the wrong path also contains a
    # missing beat. Look beyond each remaining transition for an independently
    # stable, acoustically supported future window, fit its phase, and project
    # that phase backwards until it reconnects with a compatible past run.
    # This pass may therefore change the number of events in the suspect span.
    future_window_seconds = 4.0
    maximum_future_window_seconds = 5.2
    maximum_backtrack_seconds = 8.0

    def is_unstable(item: PhaseGridEvent) -> bool:
        return (
            item.event_source in {"phase_bridge", "bidirectional_refined"}
            or item.transition_type
            in {"phase_bridge", "scale_switch", "bidirectional_bridge"}
        )

    def fit_stable_future(
        segment: list[PhaseGridEvent], start: int
    ) -> tuple[int, float, float, float] | None:
        if start >= len(segment):
            return None
        stop = start + 1
        while stop < len(segment):
            if is_unstable(segment[stop]):
                return None
            span = segment[stop].beat_time_seconds - segment[start].beat_time_seconds
            stop += 1
            if span >= future_window_seconds:
                break
            if span > maximum_future_window_seconds:
                return None
        if stop - start < 9:
            return None
        future = segment[start:stop]
        times = np.asarray([item.beat_time_seconds for item in future], dtype=float)
        if times[-1] - times[0] < future_window_seconds:
            return None
        positions = np.arange(len(times), dtype=float)
        period, phase_at_start = np.polyfit(positions, times, 1)
        period = float(period)
        phase_at_start = float(phase_at_start)
        if not minimum_period <= period <= maximum_period:
            return None
        residuals = times - (phase_at_start + positions * period)
        intervals = np.diff(times)
        if (
            float(np.max(np.abs(residuals))) > max(0.035, 0.12 * period)
            or float(np.max(np.abs(intervals - period) / period)) > 0.16
            or _robust_stability(intervals) < 0.75
        ):
            return None
        acoustic_support = float(
            np.mean(
                [
                    item.event_source == "acoustic_peak"
                    and item.beat_probability >= 0.45
                    for item in future
                ]
            )
        )
        if acoustic_support < 0.65:
            return None
        future_scales = {item.selected_scale for item in future}
        if len(future_scales) != 1:
            return None
        return stop, period, phase_at_start, _robust_stability(intervals)

    def lattice_residual(time_seconds: float, phase: float, period: float) -> float:
        step = round((time_seconds - phase) / period)
        return abs(time_seconds - (phase + step * period))

    rebuilt: list[PhaseGridEvent] = []
    segment_ids = sorted({item.activity_segment_id for item in refined})
    for segment_id in segment_ids:
        segment = [
            item for item in refined if item.activity_segment_id == segment_id
        ]
        for _ in range(6):
            unstable_positions = [
                index for index, item in enumerate(segment) if is_unstable(item)
            ]
            if not unstable_positions:
                break
            runs = np.split(
                np.asarray(unstable_positions, dtype=int),
                np.flatnonzero(np.diff(unstable_positions) > 1) + 1,
            )
            repaired_run = False
            for run in reversed(runs):
                if not len(run):
                    continue
                run_start = int(run[0])
                run_end = int(run[-1])
                future_start = run_end + 1
                future_fit = fit_stable_future(segment, future_start)
                if future_fit is None:
                    continue
                _, future_period, future_phase, future_stability = (
                    future_fit
                )

                past_candidates = [
                    index
                    for index in range(run_start)
                    if segment[run_start].beat_time_seconds
                    - segment[index].beat_time_seconds
                    <= maximum_backtrack_seconds
                ]
                if len(past_candidates) < 6:
                    continue
                # Use the decoder's intended periods here. The observed past
                # intervals are precisely what may have been stretched while a
                # beat was missing, so using them would veto the needed repair.
                past_periods = np.asarray(
                    [
                        segment[index].target_period_seconds
                        for index in past_candidates
                    ],
                    dtype=float,
                )
                recent_past_periods = past_periods[-min(len(past_periods), 12) :]
                past_period = float(np.median(recent_past_periods))
                if (
                    _robust_stability(recent_past_periods) < 0.65
                    or abs(past_period - future_period) / future_period > 0.06
                ):
                    continue

                compatibility_limit = max(0.045, 0.20 * future_period)
                anchor: int | None = None
                for candidate in reversed(past_candidates):
                    if candidate < 2:
                        continue
                    confirmation = segment[candidate - 2 : candidate + 1]
                    confirmation_times = np.asarray(
                        [item.beat_time_seconds for item in confirmation], dtype=float
                    )
                    if any(
                        lattice_residual(time_seconds, future_phase, future_period)
                        > compatibility_limit
                        for time_seconds in confirmation_times
                    ):
                        continue
                    confirmation_intervals = np.diff(confirmation_times)
                    if np.max(
                        np.abs(confirmation_intervals - future_period)
                        / future_period
                    ) > 0.10:
                        continue
                    anchor = candidate
                    break
                if anchor is None:
                    continue

                old_middle = segment[anchor + 1 : future_start]
                if len(old_middle) < 3:
                    continue
                old_residuals = np.asarray(
                    [
                        lattice_residual(
                            item.beat_time_seconds, future_phase, future_period
                        )
                        for item in old_middle
                    ],
                    dtype=float,
                )
                interval_count = int(
                    round(
                        (
                            segment[future_start].beat_time_seconds
                            - segment[anchor].beat_time_seconds
                        )
                        / future_period
                    )
                )
                if interval_count < 2:
                    continue
                expected_middle_count = interval_count - 1
                clearly_misphased = (
                    int(np.sum(old_residuals > 0.25 * future_period)) >= 3
                    and float(np.max(old_residuals)) > 0.35 * future_period
                )
                count_mismatch = expected_middle_count != len(old_middle)
                if not clearly_misphased and not count_mismatch:
                    continue

                left_time = segment[anchor].beat_time_seconds
                right_time = segment[future_start].beat_time_seconds
                corrected_period = (right_time - left_time) / interval_count
                if (
                    not minimum_period <= corrected_period <= maximum_period
                    or abs(corrected_period - future_period) / future_period > 0.06
                ):
                    continue
                projected_times = left_time + corrected_period * np.arange(
                    1, interval_count, dtype=float
                )
                templates = old_middle or [segment[anchor], segment[future_start]]
                replacement: list[PhaseGridEvent] = []
                for projected_time in projected_times:
                    template = min(
                        templates,
                        key=lambda item: abs(
                            item.beat_time_seconds - float(projected_time)
                        ),
                    )
                    beat_prob, downbeat_prob = probabilities(float(projected_time))
                    replacement.append(
                        PhaseGridEvent(
                            grid_beat_index=template.grid_beat_index,
                            activity_segment_id=segment_id,
                            beat_time_seconds=float(projected_time),
                            selected_scale=segment[future_start].selected_scale,
                            target_period_seconds=float(corrected_period),
                            predicted_time_seconds=float(projected_time),
                            phase_residual_seconds=0.0,
                            beat_probability=beat_prob,
                            downbeat_probability=downbeat_prob,
                            event_source="future_confirmed_backtrack",
                            transition_type="future_backtrack",
                            path_cost=template.path_cost,
                        )
                    )

                nearest_shifts = [
                    min(
                        abs(item.beat_time_seconds - float(projected_time))
                        for projected_time in projected_times
                    )
                    for item in old_middle
                ]
                refined_transitions.append(
                    GridTransition(
                        transition_id=len(refined_transitions) + 1,
                        activity_segment_id=segment_id,
                        start_seconds=left_time,
                        end_seconds=right_time,
                        previous_scale=segment[anchor].selected_scale,
                        next_scale=segment[future_start].selected_scale,
                        previous_period_seconds=past_period,
                        next_period_seconds=float(corrected_period),
                        phase_adjustment_seconds=(
                            float(max(nearest_shifts)) if nearest_shifts else 0.0
                        ),
                        transition_type="future_confirmed_backtrack",
                        diagnostic_note=(
                            "Stable future phase was projected backwards; "
                            f"future_stability={future_stability:.3f}, "
                            f"events={len(old_middle)}->{len(replacement)}"
                        ),
                    )
                )
                segment = (
                    segment[: anchor + 1]
                    + replacement
                    + segment[future_start:]
                )
                repaired_run = True
                break
            if not repaired_run:
                break
        rebuilt.extend(segment)
    refined = rebuilt
    refined, refined_transitions = reconcile_anchor_beat_counts(
        refined,
        refined_transitions,
        frames,
    )
    for grid_index, item in enumerate(refined, start=1):
        item.grid_beat_index = grid_index

    output_times = np.asarray([item.beat_time_seconds for item in refined], dtype=float)
    if len(output_times) > 1 and np.any(np.diff(output_times) <= 0):
        raise ValueError("Bidirectional phase refinement broke time ordering")
    output_downbeats = np.asarray(
        [
            item.beat_time_seconds
            for item in refined
            if item.downbeat_probability >= 0.5
            or (
                len(original_downbeats)
                and np.min(
                    np.abs(original_downbeats - item.beat_time_seconds)
                )
                <= 0.07
            )
        ],
        dtype=float,
    )
    cumulative_cost = 0.0
    for index, item in enumerate(refined):
        if index:
            interval = item.beat_time_seconds - refined[index - 1].beat_time_seconds
            if item.activity_segment_id == refined[index - 1].activity_segment_id:
                if not minimum_period - 1e-6 <= interval <= maximum_period + 1e-6:
                    raise ValueError("Bidirectional refinement produced invalid BPM")
                cumulative_cost += abs(
                    interval - item.target_period_seconds
                ) / max(item.target_period_seconds, 1e-9)
        cumulative_cost += 0.35 * (1.0 - item.beat_probability)
        item.path_cost = cumulative_cost
    result = BeatResult(
        method="beat-this-phase-aware",
        beat_times=output_times,
        downbeat_times=output_downbeats,
        note=(
            "CCB bidirectional phase grid with future-confirmed backtracking "
            "and conservative stable-anchor beat-count reconciliation"
        ),
    )
    return result, refined, refined_transitions


def phase_events_to_grid_decoding(events: list[PhaseGridEvent]) -> GridDecoding:
    """Describe the final phase-aware grid without rebuilding a legacy grid."""
    interval_midpoints: list[float] = []
    interval_durations: list[float] = []
    base_bpm: list[float] = []
    selected_scale: list[float] = []
    normalized_bpm: list[float] = []
    confidence: list[float] = []
    segment_ids: list[int] = []
    for segment_id in sorted({item.activity_segment_id for item in events}):
        segment = [
            item for item in events if item.activity_segment_id == segment_id
        ]
        for left, right in zip(segment, segment[1:]):
            duration = right.beat_time_seconds - left.beat_time_seconds
            if duration <= 0:
                continue
            scale = float(right.selected_scale)
            bpm = 60.0 / duration
            interval_midpoints.append(
                0.5 * (left.beat_time_seconds + right.beat_time_seconds)
            )
            interval_durations.append(duration)
            normalized_bpm.append(bpm)
            selected_scale.append(scale)
            base_bpm.append(bpm / max(scale, 1e-9))
            confidence.append(
                float(
                    np.clip(
                        0.5 * (left.beat_probability + right.beat_probability),
                        0.0,
                        1.0,
                    )
                )
            )
            segment_ids.append(segment_id)
    return GridDecoding(
        interval_midpoints=np.asarray(interval_midpoints, dtype=float),
        interval_durations=np.asarray(interval_durations, dtype=float),
        base_bpm=np.asarray(base_bpm, dtype=float),
        selected_scale=np.asarray(selected_scale, dtype=float),
        normalized_bpm=np.asarray(normalized_bpm, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
        segment_id=np.asarray(segment_ids, dtype=int),
    )


def validate_phase_grid(
    result: BeatResult,
    events: list[PhaseGridEvent],
    active_ranges: list[tuple[float, float]],
) -> list[str]:
    """Return C4 continuity errors for one final phase-aware grid."""
    errors: list[str] = []
    times = np.asarray(result.beat_times, dtype=float)
    if len(times) != len(events):
        errors.append(f"result/event count mismatch: {len(times)} != {len(events)}")
        return errors
    if len(times) and not np.all(np.isfinite(times)):
        errors.append("non-finite beat timestamp")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        errors.append("beat timestamps are not strictly increasing")
    event_times = np.asarray(
        [item.beat_time_seconds for item in events], dtype=float
    )
    if len(times) and not np.allclose(times, event_times, atol=1e-7, rtol=0.0):
        errors.append("result timestamps differ from phase event timestamps")
    if [item.grid_beat_index for item in events] != list(
        range(1, len(events) + 1)
    ):
        errors.append("phase event indices are not consecutive")
    if any(item.selected_scale not in GRID_SCALES for item in events):
        errors.append("phase event uses a scale outside the five-state grid")

    covered = np.zeros(len(times), dtype=bool)
    minimum_period = 60.0 / NORMALIZED_BPM_MAX + 1e-4
    maximum_period = 60.0 / NORMALIZED_BPM_MIN
    for segment_id, (start, end) in enumerate(active_ranges):
        is_last = segment_id == len(active_ranges) - 1
        mask = (times >= start) & (
            (times <= end) if is_last else (times < end)
        )
        covered |= mask
        segment_times = times[mask]
        segment_events = [
            item for item in events if item.activity_segment_id == segment_id
        ]
        if len(segment_times) != len(segment_events):
            errors.append(f"segment {segment_id} result/event membership mismatch")
            continue
        intervals = np.diff(segment_times)
        if np.any(intervals < minimum_period - 1e-6) or np.any(
            intervals > maximum_period + 1e-6
        ):
            errors.append(f"segment {segment_id} contains BPM outside [120, 240)")
        if len(intervals) > 1:
            ratios = intervals[1:] / intervals[:-1]
            if np.any(ratios < 0.85 - 1e-6) or np.any(ratios > 1.15 + 1e-6):
                errors.append(
                    f"segment {segment_id} contains a discontinuous interval jump"
                )
    if len(times) and not np.all(covered):
        errors.append("beat timestamp falls outside every beat-active segment")
    return errors


def finalize_phase_grid(
    greedy_result: BeatResult,
    greedy_events: list[PhaseGridEvent],
    greedy_transitions: list[GridTransition],
    frames: FramePredictions,
    active_ranges: list[tuple[float, float]],
    refinement: str,
) -> tuple[
    BeatResult,
    list[PhaseGridEvent],
    list[GridTransition],
    bool,
    bool,
    str,
]:
    """Apply optional C3 refinement, validate it, and fall back to C2 safely."""
    greedy_errors = validate_phase_grid(
        greedy_result,
        greedy_events,
        active_ranges,
    )
    if greedy_errors:
        raise ValueError(
            "C4 greedy phase grid validation failed: " + "; ".join(greedy_errors)
        )
    if refinement != "bidirectional":
        return (
            BeatResult(
                method="beat-this-phase-aware",
                beat_times=greedy_result.beat_times.copy(),
                downbeat_times=(
                    greedy_result.downbeat_times.copy()
                    if greedy_result.downbeat_times is not None
                    else np.asarray([], dtype=float)
                ),
                note="Validated greedy phase-grid fallback",
            ),
            [PhaseGridEvent(**asdict(item)) for item in greedy_events],
            [GridTransition(**asdict(item)) for item in greedy_transitions],
            False,
            False,
            "",
        )
    try:
        result, events, transitions = refine_phase_grid_bidirectionally(
            greedy_result,
            greedy_events,
            greedy_transitions,
            frames,
        )
        errors = validate_phase_grid(result, events, active_ranges)
        if errors:
            raise ValueError("; ".join(errors))
        return result, events, transitions, True, False, ""
    except ValueError as exc:
        reason = str(exc)
        return (
            BeatResult(
                method="beat-this-phase-aware",
                beat_times=greedy_result.beat_times.copy(),
                downbeat_times=(
                    greedy_result.downbeat_times.copy()
                    if greedy_result.downbeat_times is not None
                    else np.asarray([], dtype=float)
                ),
                note=(
                    "CCB fell back to the validated greedy phase grid after "
                    f"refinement validation failed: {reason}"
                ),
            ),
            [PhaseGridEvent(**asdict(item)) for item in greedy_events],
            [GridTransition(**asdict(item)) for item in greedy_transitions],
            False,
            True,
            reason,
        )


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


def audio_storage_key(path: Path) -> str:
    """Return a readable, path-unique directory key for one audio file."""
    resolved = os.path.normcase(str(Path(path).expanduser().resolve()))
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem(Path(path))}-{digest}"


def result_directory(audio_path: Path, output_root: Path) -> Path:
    """Select the hashed result directory, with matching legacy compatibility."""
    preferred = output_root / audio_storage_key(audio_path)
    if preferred.exists():
        return preferred
    legacy = output_root / safe_stem(audio_path)
    report_path = legacy / "report.json"
    if report_path.is_file():
        try:
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            if Path(report.get("audio", "")).resolve() == audio_path.resolve():
                return legacy
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return preferred


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
        cache_schema_version=int(row.get("cache_schema_version") or 1),
        audio_size_bytes=(
            int(row["audio_size_bytes"])
            if row.get("audio_size_bytes") else None
        ),
        audio_mtime_ns=(
            int(row["audio_mtime_ns"])
            if row.get("audio_mtime_ns") else None
        ),
        checkpoint=row.get("checkpoint") or None,
        beat_this_version=row.get("beat_this_version") or None,
        ccb_version=row.get("ccb_version") or None,
    )


def write_frame_predictions_csv(path: Path, frames: FramePredictions) -> None:
    lengths = {
        len(frames.fused_beat_logits),
        len(frames.fused_downbeat_logits),
    }
    if len(lengths) != 1:
        raise ValueError(f"Frame prediction lengths differ: {sorted(lengths)}")

    fused_beat_prob = expit(frames.fused_beat_logits)
    fused_downbeat_prob = expit(frames.fused_downbeat_logits)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "time_seconds",
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


def build_reliability_segments(
    phase_events: list[PhaseGridEvent],
    transitions: list[GridTransition],
    diagnostics: list[IntervalDiagnostic],
    active_ranges: list[tuple[float, float]],
    blocked_ranges: list[tuple[float, float]],
    activity_segments: list[ActivitySegment] | None = None,
    bin_seconds: float = 2.0,
) -> list[ReliabilitySegment]:
    """Describe where the final grid is trustworthy without changing it.

    Reliability consumes phase events, transitions, and interval diagnostics.
    Its labels are an audit aid, not another beat-repair stage.
    """
    interval_midpoints: list[float] = []
    interval_bpms: list[float] = []
    for activity_id in sorted({item.activity_segment_id for item in phase_events}):
        times = np.asarray(
            [
                item.beat_time_seconds
                for item in phase_events
                if item.activity_segment_id == activity_id
            ],
            dtype=float,
        )
        if len(times) < 2:
            continue
        intervals = np.diff(times)
        interval_midpoints.extend((times[:-1] + intervals / 2.0).tolist())
        interval_bpms.extend((60.0 / intervals).tolist())
    reference_bpm = modal_tempo(
        np.asarray(interval_midpoints, dtype=float),
        np.asarray(interval_bpms, dtype=float),
        NORMALIZED_BPM_MIN,
        NORMALIZED_BPM_MAX,
    )

    bins: list[ReliabilitySegment] = []
    repair_sources = {
        "bidirectional_refined",
        "future_confirmed_backtrack",
        "future_phase_start",
        "anchor_count_reconciled",
    }
    repair_transitions = {
        "bidirectional_bridge",
        "future_confirmed_backtrack",
        "future_phase_start",
        "anchor_count_reconciled",
    }
    for activity_id, (range_start, range_end) in enumerate(active_ranges):
        local_events = [
            item
            for item in phase_events
            if item.activity_segment_id == activity_id
        ]
        event_times = np.asarray(
            [item.beat_time_seconds for item in local_events], dtype=float
        )
        if len(event_times) >= 2:
            periods = np.diff(event_times)
            period_midpoints = event_times[:-1] + periods / 2.0
        else:
            periods = np.asarray([], dtype=float)
            period_midpoints = np.asarray([], dtype=float)

        start = range_start
        while start < range_end - 1e-9:
            end = min(start + bin_seconds, range_end)
            event_subset = [
                item
                for item in local_events
                if start <= item.beat_time_seconds < end
            ]
            period_subset = periods[
                (period_midpoints >= start) & (period_midpoints < end)
            ]
            local_bpm = (
                float(60.0 / np.median(period_subset))
                if len(period_subset)
                else None
            )
            interval_cv = (
                float(np.std(period_subset) / np.mean(period_subset))
                if len(period_subset) >= 2 and np.mean(period_subset) > 0
                else 0.0
            )
            acoustic_support = (
                float(
                    np.mean(
                        [
                            max(item.beat_probability, item.downbeat_probability)
                            >= 0.45
                            for item in event_subset
                        ]
                    )
                )
                if event_subset
                else 0.0
            )
            local_transitions = [
                item
                for item in transitions
                if item.activity_segment_id == activity_id
                and item.end_seconds > start
                and item.start_seconds < end
            ]
            scale_switches = sum(
                item.transition_type == "scale_switch"
                for item in local_transitions
            )
            bridge_steps = sum(
                item.transition_type == "phase_bridge"
                for item in local_transitions
            )
            phase_repairs = sum(
                item.event_source in repair_sources for item in event_subset
            ) + sum(
                item.transition_type in repair_transitions
                for item in local_transitions
            )
            has_future_phase_start = any(
                item.transition_type == "future_phase_start"
                for item in local_transitions
            )
            has_anchor_count_reconciliation = any(
                item.transition_type == "anchor_count_reconciled"
                for item in local_transitions
            )
            local_diagnostics = [
                item
                for item in diagnostics
                if item.activity_segment_id == activity_id
                and start <= item.midpoint_seconds < end
            ]
            protected_ratio = (
                sum(
                    item.classification == "protected_tempo_motion"
                    for item in local_diagnostics
                )
                / len(local_diagnostics)
                if local_diagnostics
                else 0.0
            )
            outlier_ratio = (
                sum(
                    item.classification
                    in {"strong_outlier", "structural_evidence_required"}
                    for item in local_diagnostics
                )
                / len(local_diagnostics)
                if local_diagnostics
                else 0.0
            )
            bpm_deviation = (
                abs(local_bpm - reference_bpm) / reference_bpm
                if local_bpm is not None and np.isfinite(reference_bpm)
                else 0.0
            )
            stability = float(np.clip(1.0 - interval_cv / 0.12, 0.0, 1.0))
            reasons: list[str] = []

            is_tempo_motion = (
                protected_ratio >= 0.30
                and interval_cv < 0.12
                and scale_switches <= 1
            )
            structural_unreliable = (
                (
                    scale_switches >= 2
                    and (acoustic_support < 0.75 or interval_cv >= 0.08)
                )
                or interval_cv >= 0.10
                or (bridge_steps >= 2 and acoustic_support < 0.55)
            )
            evidence_unreliable = (
                (bpm_deviation >= 0.15 and acoustic_support < 0.60)
                or (outlier_ratio >= 0.45 and acoustic_support < 0.55)
            )
            unreliable = structural_unreliable or (
                evidence_unreliable
                and not is_tempo_motion
                and not has_future_phase_start
                and not has_anchor_count_reconciliation
            )
            if unreliable:
                classification = "BEAT_THIS_UNRELIABLE"
                risk = max(
                    min(1.0, scale_switches / 2.0),
                    min(1.0, interval_cv / 0.12),
                    min(1.0, bpm_deviation / 0.25)
                    * (1.0 - 0.5 * acoustic_support),
                    outlier_ratio * (1.0 - 0.4 * acoustic_support),
                )
                score = float(np.clip(0.50 * (1.0 - risk), 0.05, 0.49))
                if scale_switches:
                    reasons.append(f"{scale_switches} grid-scale switch(es)")
                if interval_cv >= 0.10:
                    reasons.append(f"interval variation {100.0 * interval_cv:.1f}%")
                if bpm_deviation >= 0.15:
                    reasons.append(
                        f"local BPM differs {100.0 * bpm_deviation:.1f}% from dominant"
                    )
                if bridge_steps >= 2:
                    reasons.append(f"{bridge_steps} phase bridges")
                if outlier_ratio >= 0.45:
                    reasons.append(f"interval outlier share {outlier_ratio:.2f}")
            elif is_tempo_motion:
                classification = "TEMPO_MOTION"
                score = float(
                    np.clip(0.55 + 0.25 * stability + 0.20 * acoustic_support, 0, 1)
                )
                reasons.append("coherent tempo motion protected")
            elif phase_repairs or bridge_steps:
                classification = "PHASE_REPAIRED"
                score = float(
                    np.clip(0.58 + 0.22 * stability + 0.20 * acoustic_support, 0, 1)
                )
                reasons.append("phase-aware refinement or bridge was used")
            else:
                classification = "RELIABLE"
                score = float(
                    np.clip(0.55 + 0.30 * stability + 0.15 * acoustic_support, 0, 1)
                )
                reasons.append("stable normalized grid")

            reasons.append(f"acoustic support {acoustic_support:.2f}")
            bins.append(
                ReliabilitySegment(
                    segment_id=0,
                    activity_segment_id=activity_id,
                    start_seconds=float(start),
                    end_seconds=float(end),
                    classification=classification,
                    reliability_score=score,
                    reference_bpm=(
                        float(reference_bpm) if np.isfinite(reference_bpm) else None
                    ),
                    local_bpm=local_bpm,
                    acoustic_support=acoustic_support,
                    scale_switches=scale_switches,
                    phase_repairs=phase_repairs,
                    reason="; ".join(reasons),
                )
            )
            start = end

    # A short non-red hole inside a longer suspect run is usually only a bin
    # boundary artefact. Close up to four seconds, but never consume coherent
    # tempo motion or cross an activity boundary.
    index = 0
    while index < len(bins):
        if bins[index].classification != "BEAT_THIS_UNRELIABLE":
            index += 1
            continue
        following = index + 1
        while (
            following < len(bins)
            and bins[following].activity_segment_id == bins[index].activity_segment_id
            and bins[following].classification != "BEAT_THIS_UNRELIABLE"
        ):
            following += 1
        gap = bins[index + 1 : following]
        if (
            following < len(bins)
            and bins[following].activity_segment_id == bins[index].activity_segment_id
            and gap
            and gap[-1].end_seconds - gap[0].start_seconds <= 4.0 + 1e-6
            and all(item.classification != "TEMPO_MOTION" for item in gap)
        ):
            for item in gap:
                item.classification = "BEAT_THIS_UNRELIABLE"
                item.reliability_score = min(item.reliability_score, 0.49)
                item.reason = "between adjacent unreliable windows; " + item.reason
        index = max(index + 1, following)

    note_segments = activity_segments or []
    for start, end in blocked_ranges:
        matching_notes = [
            item.note
            for item in note_segments
            if item.no_beat
            and item.end_seconds > start
            and item.start_seconds < end
            and item.note
        ]
        bins.append(
            ReliabilitySegment(
                segment_id=0,
                activity_segment_id=-1,
                start_seconds=float(start),
                end_seconds=float(end),
                classification="NO_BEAT",
                reliability_score=1.0,
                reference_bpm=(
                    float(reference_bpm) if np.isfinite(reference_bpm) else None
                ),
                local_bpm=None,
                acoustic_support=0.0,
                scale_switches=0,
                phase_repairs=0,
                reason=(
                    "user-marked NO_BEAT: " + " / ".join(dict.fromkeys(matching_notes))
                    if matching_notes
                    else "user-marked NO_BEAT interval"
                ),
            )
        )

    merged: list[ReliabilitySegment] = []
    for item in sorted(bins, key=lambda value: (value.start_seconds, value.end_seconds)):
        if (
            merged
            and merged[-1].classification == item.classification
            and merged[-1].activity_segment_id == item.activity_segment_id
            and abs(merged[-1].end_seconds - item.start_seconds) <= 1e-6
        ):
            previous = merged[-1]
            old_duration = previous.end_seconds - previous.start_seconds
            new_duration = item.end_seconds - item.start_seconds
            total_duration = old_duration + new_duration
            previous.end_seconds = item.end_seconds
            previous.reliability_score = (
                previous.reliability_score * old_duration
                + item.reliability_score * new_duration
            ) / total_duration
            previous.acoustic_support = (
                previous.acoustic_support * old_duration
                + item.acoustic_support * new_duration
            ) / total_duration
            if previous.local_bpm is None:
                previous.local_bpm = item.local_bpm
            elif item.local_bpm is not None:
                previous.local_bpm = (
                    previous.local_bpm * old_duration + item.local_bpm * new_duration
                ) / total_duration
            previous.scale_switches += item.scale_switches
            previous.phase_repairs += item.phase_repairs
            clauses = list(dict.fromkeys((previous.reason + "; " + item.reason).split("; ")))
            previous.reason = "; ".join(clauses[:8])
        else:
            merged.append(ReliabilitySegment(**asdict(item)))
    for segment_id, item in enumerate(merged, start=1):
        item.segment_id = segment_id
    return merged


def write_beats_csv(
    path: Path,
    result: BeatResult,
    midpoints: np.ndarray,
    raw_bpm: np.ndarray,
    smooth_bpm: np.ndarray,
    sample_rate: int,
    ranges: list[tuple[float, float]] | None = None,
    reliability: list[ReliabilitySegment] | None = None,
    manual_reasons: dict[float, str] | None = None,
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
                "reliability_class",
                "reliability_score",
                "reliability_reason",
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

        def reliability_at(time_seconds: float) -> ReliabilitySegment | None:
            if reliability is None:
                return None
            for item in reliability:
                if item.start_seconds <= time_seconds < item.end_seconds:
                    return item
            if reliability and np.isclose(time_seconds, reliability[-1].end_seconds):
                return reliability[-1]
            return None

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
            reliability_item = reliability_at(float(beat_time))
            manual_reason = next(
                (
                    reason
                    for manual_time, reason in (manual_reasons or {}).items()
                    if abs(float(beat_time) - manual_time) <= 1e-6
                ),
                None,
            )
            if manual_reason is not None:
                reliability_values = ["MANUAL_EDIT", "1.000000", manual_reason]
            else:
                reliability_values = [
                    reliability_item.classification if reliability_item else "",
                    (
                        f"{reliability_item.reliability_score:.6f}"
                        if reliability_item
                        else ""
                    ),
                    reliability_item.reason if reliability_item else "",
                ]
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
                ] + reliability_values
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
                ] + reliability_values
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


MANUAL_BEAT_MATCH_TOLERANCE = 0.02
MANUAL_BEAT_COLLISION_TOLERANCE = 1e-6


def empty_manual_beat_edits() -> dict[str, list[dict]]:
    return {"added": [], "adjusted": [], "deleted": []}


def normalize_manual_beat_edits(value: object) -> dict[str, list[dict]]:
    """Return a safe copy of persisted manual beat operations."""
    result = empty_manual_beat_edits()
    if not isinstance(value, dict):
        return result
    for key in result:
        items = value.get(key, [])
        if isinstance(items, list):
            result[key] = [dict(item) for item in items if isinstance(item, dict)]
    return result


def apply_manual_beat_edits(
    automatic: BeatResult,
    edits: dict[str, list[dict]],
    ranges: list[tuple[float, float]] | None = None,
) -> tuple[BeatResult, dict[float, str], list[str]]:
    """Apply persisted manual operations to an automatic grid."""
    times = [float(value) for value in automatic.beat_times]
    downbeats = [
        bool(
            automatic.downbeat_times is not None
            and len(automatic.downbeat_times)
            and np.min(np.abs(automatic.downbeat_times - value)) <= 0.05
        )
        for value in times
    ]
    reasons: dict[float, str] = {}
    warnings: list[str] = []

    def find_time(target: float, tolerance: float) -> int | None:
        if not times:
            return None
        distances = np.abs(np.asarray(times) - target)
        index = int(np.argmin(distances))
        return index if distances[index] <= tolerance else None

    for item in edits.get("deleted", []):
        try:
            original = float(item["original_time_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        index = find_time(original, MANUAL_BEAT_MATCH_TOLERANCE)
        if index is not None:
            times.pop(index)
            downbeats.pop(index)

    for item in edits.get("adjusted", []):
        try:
            original = float(item["original_time_seconds"])
            new_time = float(item["new_time_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        is_downbeat = bool(item.get("is_downbeat", False))
        index = find_time(original, MANUAL_BEAT_MATCH_TOLERANCE)
        if index is None:
            warnings.append(
                f"Automatic beat near {original:.9f}s was absent; kept manual beat at {new_time:.9f}s"
            )
            times.append(new_time)
            downbeats.append(is_downbeat)
        else:
            times[index] = new_time
            downbeats[index] = is_downbeat
        reasons[new_time] = "manual_adjusted"

    for item in edits.get("added", []):
        try:
            new_time = float(item["time_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        times.append(new_time)
        downbeats.append(bool(item.get("is_downbeat", False)))
        reasons[new_time] = "manual_added"

    ordered = sorted(zip(times, downbeats), key=lambda item: item[0])
    filtered: list[tuple[float, bool]] = []
    for time_seconds, is_downbeat in ordered:
        if ranges is not None and not any(
            start <= time_seconds <= end for start, end in ranges
        ):
            continue
        if filtered and abs(time_seconds - filtered[-1][0]) <= MANUAL_BEAT_COLLISION_TOLERANCE:
            raise ValueError(
                f"Manual beat edit overlaps an existing beat at {time_seconds:.9f}s"
            )
        filtered.append((time_seconds, is_downbeat))
    final_times = np.asarray([item[0] for item in filtered], dtype=float)
    final_downbeats = np.asarray(
        [item[0] for item in filtered if item[1]], dtype=float
    )
    return (
        BeatResult(
            method="ccb-final",
            beat_times=final_times,
            downbeat_times=final_downbeats,
            note="Validated phase-aware CCB beat grid with manual edits",
        ),
        reasons,
        warnings,
    )


def validate_mix_gains(
    music_gain: float,
    click_gain: float,
) -> tuple[float, float]:
    """Validate independent music/click amplitude gains."""
    try:
        music = float(music_gain)
        click = float(click_gain)
    except (TypeError, ValueError) as exc:
        raise ValueError("music_gain and click_gain must be finite numbers") from exc
    if not math.isfinite(music) or not math.isfinite(click):
        raise ValueError("music_gain and click_gain must be finite numbers")
    if music < 0 or click < 0:
        raise ValueError("music_gain and click_gain must be non-negative")
    if music == 0 and click == 0:
        raise ValueError("music_gain and click_gain cannot both be zero")
    return music, click


def write_click_track(
    path: Path,
    y: np.ndarray,
    sr: int,
    result: BeatResult,
    *,
    music_gain: float = DEFAULT_MUSIC_GAIN,
    click_gain: float = DEFAULT_CLICK_GAIN,
) -> None:
    music_gain, click_gain = validate_mix_gains(music_gain, click_gain)
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
    mix = (
        music_gain * librosa.util.normalize(y)
        + click_gain * librosa.util.normalize(clicks)
    )
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    sf.write(path, mix, sr, subtype="PCM_16")


def write_overview_plot(
    path: Path,
    title: str,
    frames: FramePredictions,
    fused_result: BeatResult,
    phase_result: BeatResult,
    phase_events: list[PhaseGridEvent],
    decoding: GridDecoding,
    reliability: list[ReliabilitySegment],
    duration: float,
) -> None:
    """Render final beat evidence, tempo, grid scale, and review ranges."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 11.5), sharex=True)
    colours = {
        "RELIABLE": "tab:green",
        "PHASE_REPAIRED": "tab:blue",
        "TEMPO_MOTION": "tab:orange",
        "NO_BEAT": "0.45",
        "BEAT_THIS_UNRELIABLE": "tab:red",
    }
    alphas = {
        "RELIABLE": 0.025,
        "PHASE_REPAIRED": 0.09,
        "TEMPO_MOTION": 0.11,
        "NO_BEAT": 0.16,
        "BEAT_THIS_UNRELIABLE": 0.15,
    }
    shown: set[str] = set()
    for item in reliability:
        for ax in axes:
            ax.axvspan(
                item.start_seconds,
                item.end_seconds,
                color=colours[item.classification],
                alpha=alphas[item.classification],
                label=(
                    item.classification
                    if item.classification not in shown and ax is axes[0]
                    else None
                ),
            )
        shown.add(item.classification)

    frame_times = np.arange(len(frames.fused_beat_logits), dtype=float) / frames.fps
    axes[0].plot(
        frame_times,
        expit(frames.fused_beat_logits),
        color="tab:blue",
        linewidth=0.7,
        alpha=0.8,
        label="fused beat probability",
    )
    axes[0].plot(
        frame_times,
        expit(frames.fused_downbeat_logits),
        color="tab:purple",
        linewidth=0.6,
        alpha=0.55,
        label="fused downbeat probability",
    )
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Probability")

    axes[1].scatter(
        fused_result.beat_times,
        np.ones(len(fused_result.beat_times)),
        marker="|",
        s=42,
        color="0.25",
        alpha=0.6,
        label="Beat This! fused",
    )
    ordinary_times = np.asarray(phase_result.beat_times, dtype=float)
    repaired_times = np.asarray(
        [
            item.beat_time_seconds
            for item in phase_events
            if item.event_source
            in {
                "bidirectional_refined",
                "future_confirmed_backtrack",
                "future_phase_start",
                "anchor_count_reconciled",
            }
            and np.any(
                np.abs(np.asarray(phase_result.beat_times) - item.beat_time_seconds)
                <= 1e-6
            )
        ],
        dtype=float,
    )
    axes[1].scatter(
        ordinary_times,
        np.zeros(len(ordinary_times)),
        marker="|",
        s=48,
        color="tab:green",
        alpha=0.7,
        label="final phase grid",
    )
    if len(repaired_times):
        axes[1].scatter(
            repaired_times,
            np.zeros(len(repaired_times)),
            marker="|",
            s=65,
            color="tab:blue",
            label="phase-refined event",
        )
    axes[1].set_yticks([0, 1], ["final grid", "fused"])
    axes[1].set_ylim(-0.55, 1.55)

    bpm_label_used = False
    reference_values = [
        item.reference_bpm for item in reliability if item.reference_bpm is not None
    ]
    for activity_id in sorted({item.activity_segment_id for item in phase_events}):
        times = np.asarray(
            [
                item.beat_time_seconds
                for item in phase_events
                if item.activity_segment_id == activity_id
            ],
            dtype=float,
        )
        if len(times) < 2:
            continue
        periods = np.diff(times)
        axes[2].plot(
            times[:-1] + periods / 2.0,
            60.0 / periods,
            color="tab:green",
            linewidth=0.85,
            alpha=0.78,
            label="final interval BPM" if not bpm_label_used else None,
        )
        bpm_label_used = True
    if reference_values:
        axes[2].axhline(
            float(np.median(reference_values)),
            color="black",
            linestyle="--",
            linewidth=0.9,
            label=f"dominant {np.median(reference_values):.1f} BPM",
        )
    axes[2].axhspan(
        NORMALIZED_BPM_MIN,
        NORMALIZED_BPM_MAX,
        color="tab:green",
        alpha=0.035,
    )
    axes[2].set_ylim(110, 250)
    axes[2].set_ylabel("BPM")

    if len(decoding.interval_midpoints):
        axes[3].step(
            decoding.interval_midpoints,
            decoding.selected_scale,
            where="mid",
            color="tab:blue",
            linewidth=1.0,
            label="selected grid scale",
        )
    axes[3].set_yticks(GRID_SCALES, [f"{scale:g}x" for scale in GRID_SCALES])
    axes[3].set_ylabel("Grid scale")
    axes[3].set_xlabel("Time (seconds)")

    for ax in axes:
        ax.set_xlim(0, duration)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle(f"{title} — CCB result overview")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def default_cache_root() -> Path:
    """Return CCB's per-user cache root, honoring ``CCB_CACHE_DIR``."""
    configured = os.environ.get("CCB_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(user_cache_dir("CCB"))


def legacy_cache_root(output_root: Path) -> Path:
    """Return the pre-package cache root used by older CCB checkouts."""
    return output_root.parent / ".ccb-cache"


def beat_this_version() -> str:
    try:
        return importlib.metadata.version("beat-this")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def inference_cache_paths(
    audio_path: Path,
    output_root: Path,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    if cache_dir is not None:
        cache_roots = [Path(cache_dir).expanduser()]
    else:
        cache_roots = [default_cache_root()]
        if "CCB_CACHE_DIR" not in os.environ:
            cache_roots.append(legacy_cache_root(output_root))
    preferred = cache_roots[0] / audio_storage_key(audio_path)
    if preferred.exists():
        cache_output = preferred
    else:
        cache_output = preferred
        for cache_root in cache_roots:
            legacy_output = cache_root / safe_stem(audio_path)
            metadata_path = legacy_output / "inference.csv"
            if not metadata_path.is_file():
                continue
            try:
                metadata = read_inference_metadata_csv(metadata_path)
                if Path(metadata.audio_path).resolve() == audio_path.resolve():
                    cache_output = legacy_output
                    break
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return {
        "root": cache_output,
        "metadata": cache_output / "inference.csv",
        "frames": cache_output / "frames.csv",
        "fused": cache_output / "fused_beats.csv",
    }


def inference_cache_available(
    audio_path: Path,
    output_root: Path,
    cache_dir: Path | None = None,
    *,
    sample_rate: int | None = None,
    checkpoint: str | None = None,
    hop_seconds: float | None = None,
) -> bool:
    paths = inference_cache_paths(audio_path, output_root, cache_dir)
    if not all(paths[name].is_file() for name in ("metadata", "frames", "fused")):
        return False
    try:
        metadata = read_inference_metadata_csv(paths["metadata"])
        audio_stat = audio_path.stat()
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if Path(metadata.audio_path).resolve() != audio_path.resolve():
        return False
    if metadata.cache_schema_version > CACHE_SCHEMA_VERSION:
        return False
    if sample_rate is not None and metadata.sample_rate != sample_rate:
        return False
    if hop_seconds is not None and not math.isclose(
        metadata.hop_seconds, hop_seconds, abs_tol=1e-9
    ):
        return False
    if metadata.audio_size_bytes is not None and (
        metadata.audio_size_bytes != audio_stat.st_size
    ):
        return False
    if metadata.audio_mtime_ns is not None and (
        metadata.audio_mtime_ns != audio_stat.st_mtime_ns
    ):
        return False
    if checkpoint is not None and metadata.checkpoint is not None and (
        metadata.checkpoint != checkpoint
    ):
        return False
    current_beat_this = beat_this_version()
    if metadata.beat_this_version not in {None, "unknown"} and (
        metadata.beat_this_version != current_beat_this
    ):
        return False
    return True


def analyse_file(
    audio_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    estimator: BeatThisEstimator | None,
) -> list[dict]:
    """Run the sole production pipeline and emit only final user artifacts."""
    y, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    file_output = result_directory(audio_path, output_root)
    file_output.mkdir(parents=True, exist_ok=True)
    user_paths = {
        "beats": file_output / "beats.csv",
        "click": file_output / "click.wav",
        "overview": file_output / "overview.png",
        "segments": file_output / "segments.csv",
        "report": file_output / "report.json",
    }
    cache_paths = inference_cache_paths(
        audio_path,
        output_root,
        getattr(args, "cache_dir", None),
    )
    cache_paths["root"].mkdir(parents=True, exist_ok=True)

    manual_edits = empty_manual_beat_edits()
    if getattr(args, "preserve_manual_edits", True) and user_paths["report"].is_file():
        try:
            with user_paths["report"].open(encoding="utf-8") as handle:
                previous_report = json.load(handle)
            if Path(previous_report.get("audio", "")).resolve() == audio_path.resolve():
                manual_edits = normalize_manual_beat_edits(
                    previous_report.get("manual_edits")
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            manual_edits = empty_manual_beat_edits()

    ensure_segments_csv(user_paths["segments"], duration)
    activity_segments = read_segments_csv(user_paths["segments"], duration)
    ranges = active_ranges(activity_segments, duration)
    blocked_ranges = no_beat_ranges(activity_segments, duration)

    cache_ready = inference_cache_available(
        audio_path,
        output_root,
        getattr(args, "cache_dir", None),
        sample_rate=args.sample_rate,
        checkpoint=args.beat_this_checkpoint,
        hop_seconds=args.beat_this_hop_seconds,
    )
    if args.refresh_cache or not cache_ready:
        if estimator is None:
            raise FileNotFoundError(
                "Inference cache is missing or refresh was requested, but Beat This! "
                "was not initialized."
            )
        fused_result, inferred_frames = estimator(y, sr)
        fused_result.beat_times = np.unique(
            fused_result.beat_times[np.isfinite(fused_result.beat_times)]
        )
        audio_stat = audio_path.stat()
        metadata = InferenceMetadata(
            fps=inferred_frames.fps,
            window_seconds=inferred_frames.window_seconds,
            hop_seconds=inferred_frames.hop_seconds,
            overlap_windows=inferred_frames.overlap_windows,
            sample_rate=sr,
            duration_seconds=duration,
            audio_path=str(audio_path.resolve()),
            cache_schema_version=CACHE_SCHEMA_VERSION,
            audio_size_bytes=audio_stat.st_size,
            audio_mtime_ns=audio_stat.st_mtime_ns,
            checkpoint=args.beat_this_checkpoint,
            beat_this_version=beat_this_version(),
            ccb_version=CCB_VERSION,
        )
        write_inference_metadata_csv(cache_paths["metadata"], metadata)
        write_frame_predictions_csv(cache_paths["frames"], inferred_frames)
        fused_midpoints, fused_raw, fused_smooth = local_tempo(
            fused_result.beat_times
        )
        write_beats_csv(
            cache_paths["fused"],
            fused_result,
            fused_midpoints,
            fused_raw,
            fused_smooth,
            sr,
        )

    metadata = read_inference_metadata_csv(cache_paths["metadata"])
    if metadata.sample_rate != sr:
        raise ValueError(
            f"Cached sample rate {metadata.sample_rate} does not match loaded rate {sr}; "
            "rerun with --refresh-cache."
        )
    if abs(metadata.duration_seconds - duration) > 0.05:
        raise ValueError(
            "Cached inference duration does not match the audio; "
            "rerun with --refresh-cache."
        )
    if Path(metadata.audio_path).resolve() != audio_path.resolve():
        raise ValueError(
            "Cached inference belongs to a different audio path; "
            "rerun with --refresh-cache."
        )
    frames = read_frame_predictions_csv(cache_paths["frames"], metadata)
    fused_result = read_beats_csv(
        cache_paths["fused"],
        "beat-this-fused",
        "Beat This! overlap-fused inference",
    )

    interval_diagnostics = diagnose_fused_beats(
        fused_result,
        frames,
        ranges,
    )
    greedy_result, greedy_events, greedy_transitions = build_phase_aware_grid(
        fused_result,
        frames,
        ranges,
    )
    greedy_result.method = "ccb-greedy-fallback"
    (
        phase_result,
        phase_events,
        grid_transitions,
        refinement_applied,
        fallback_used,
        fallback_reason,
    ) = finalize_phase_grid(
        greedy_result,
        greedy_events,
        greedy_transitions,
        frames,
        ranges,
        "bidirectional",
    )
    validation_errors = validate_phase_grid(phase_result, phase_events, ranges)
    if validation_errors:
        raise ValueError(
            "Final phase grid validation failed: " + "; ".join(validation_errors)
        )
    grid_decoding = phase_events_to_grid_decoding(phase_events)
    reliability_segments = build_reliability_segments(
        phase_events,
        grid_transitions,
        interval_diagnostics,
        ranges,
        blocked_ranges,
        activity_segments,
    )
    final_result = BeatResult(
        method="ccb-final",
        beat_times=phase_result.beat_times.copy(),
        downbeat_times=(
            phase_result.downbeat_times.copy()
            if phase_result.downbeat_times is not None
            else np.asarray([], dtype=float)
        ),
        note="Validated phase-aware CCB beat grid",
    )
    final_result, manual_reasons, manual_warnings = apply_manual_beat_edits(
        final_result,
        manual_edits,
        ranges,
    )
    final_midpoints, final_raw, final_smooth = local_tempo(final_result.beat_times)
    write_beats_csv(
        user_paths["beats"],
        final_result,
        final_midpoints,
        final_raw,
        final_smooth,
        sr,
        ranges,
        reliability_segments,
        manual_reasons,
    )
    if args.no_click:
        user_paths["click"].unlink(missing_ok=True)
    else:
        write_click_track(
            user_paths["click"],
            y,
            sr,
            final_result,
            music_gain=args.music_gain,
            click_gain=args.click_gain,
        )
    write_overview_plot(
        user_paths["overview"],
        audio_path.name,
        frames,
        fused_result,
        final_result,
        phase_events,
        grid_decoding,
        reliability_segments,
        duration,
    )

    analysis = analyse_result_from_ranges(
        final_result,
        ranges,
        duration,
        args,
        audio_path,
    )
    reliability_classes = (
        "RELIABLE",
        "PHASE_REPAIRED",
        "TEMPO_MOTION",
        "NO_BEAT",
        "BEAT_THIS_UNRELIABLE",
    )
    report = {
        "schema_version": 1,
        "audio": str(audio_path.resolve()),
        "duration_seconds": round(duration, 3),
        "result": {
            "beats_csv": user_paths["beats"].name,
            "click_wav": None if args.no_click else user_paths["click"].name,
            "overview_png": user_paths["overview"].name,
            "segments_csv": user_paths["segments"].name,
            "beat_count": int(len(final_result.beat_times)),
            "downbeat_count": int(len(final_result.downbeat_times)),
            "dominant_bpm": (
                round(float(analysis["main_bpm"]), 3)
                if np.isfinite(analysis["main_bpm"])
                else None
            ),
            "target_bpm_interval": "[120, 240)",
            "candidate_scales": [float(scale) for scale in GRID_SCALES],
        },
        "activity": {
            "active_seconds": round(sum(end - start for start, end in ranges), 3),
            "no_beat_seconds": round(
                sum(end - start for start, end in blocked_ranges), 3
            ),
            "no_beat_ranges": [
                {"start_seconds": start, "end_seconds": end}
                for start, end in blocked_ranges
            ],
        },
        "manual_edits": manual_edits,
        "manual_edit_warnings": manual_warnings,
        "reliability": {
            "classification_counts": {
                name: sum(
                    item.classification == name for item in reliability_segments
                )
                for name in reliability_classes
            },
            "classification_seconds": {
                name: round(
                    sum(
                        item.end_seconds - item.start_seconds
                        for item in reliability_segments
                        if item.classification == name
                    ),
                    3,
                )
                for name in reliability_classes
            },
            "recommended_review_ranges": [
                {
                    "start_seconds": round(item.start_seconds, 3),
                    "end_seconds": round(item.end_seconds, 3),
                    "classification": item.classification,
                    "reliability_score": round(item.reliability_score, 3),
                    "reason": item.reason,
                }
                for item in reliability_segments
                if item.classification
                in {"BEAT_THIS_UNRELIABLE", "TEMPO_MOTION"}
            ],
        },
        "processing": {
            "model": "Beat This!",
            "cache_directory": str(cache_paths["root"].resolve()),
            "click_mix": {
                "music_gain": args.music_gain,
                "click_gain": args.click_gain,
            },
            "phase_refinement": "bidirectional",
            "refinement_applied": refinement_applied,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason or None,
        },
    }
    with user_paths["report"].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    os.utime(cache_paths["root"], None)
    return [analysis]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a final CCB beat grid, click track, overview, and report."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CCB {CCB_VERSION}",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Audio file(s): WAV/MP3/FLAC/M4A...")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"))
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
        "--refresh-cache",
        action="store_true",
        help="Rerun Beat This! instead of using the hidden inference cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Inference cache directory (default: CCB_CACHE_DIR or the "
            "operating-system user cache directory)"
        ),
    )
    parser.add_argument(
        "--no-click",
        action="store_true",
        help="Do not create the final click-track WAV",
    )
    parser.set_defaults(
        music_gain=DEFAULT_MUSIC_GAIN,
        click_gain=DEFAULT_CLICK_GAIN,
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
    needs_inference = args.refresh_cache or any(
        not inference_cache_available(
            audio_path,
            args.output,
            args.cache_dir,
            sample_rate=args.sample_rate,
            checkpoint=args.beat_this_checkpoint,
            hop_seconds=args.beat_this_hop_seconds,
        )
        for audio_path in args.audio
    )
    if needs_inference:
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
            all_analyses.extend(
                analyse_file(audio_path, args.output, args, beat_this)
            )
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
                    "ccb-final",
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
