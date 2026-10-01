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
class TempoRegion:
    start_seconds: float
    end_seconds: float
    median_bpm: float
    difference_from_main_bpm: float
    beat_intervals: int


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


def modal_tempo(midpoints: np.ndarray, bpm: np.ndarray, min_bpm: float, max_bpm: float) -> float:
    """Find the time-weighted dominant tempo rather than merely the median."""
    valid = np.isfinite(bpm) & (bpm >= min_bpm) & (bpm <= max_bpm)
    if not np.any(valid):
        return math.nan
    values = bpm[valid]
    valid_times = midpoints[valid]
    if len(valid_times) > 1:
        weights = np.gradient(valid_times)
        weights = np.clip(weights, 1e-3, None)
    else:
        weights = np.ones_like(values)

    edges = np.arange(math.floor(min_bpm), math.ceil(max_bpm) + 1.0, 1.0)
    hist, _ = np.histogram(values, bins=edges, weights=weights)
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


def safe_stem(path: Path) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", path.stem)
    return cleaned.strip(" ._") or "audio"


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
                    f"{index / frames.fps:.6f}",
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


def write_beats_csv(
    path: Path,
    result: BeatResult,
    midpoints: np.ndarray,
    raw_bpm: np.ndarray,
    smooth_bpm: np.ndarray,
) -> None:
    downbeats = np.asarray(result.downbeat_times if result.downbeat_times is not None else [])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "beat_index",
                "beat_time_seconds",
                "is_downbeat",
                "interval_midpoint_seconds",
                "raw_local_bpm",
                "smoothed_local_bpm",
            ]
        )
        for i, beat_time in enumerate(result.beat_times):
            is_downbeat = bool(
                downbeats.size and np.min(np.abs(downbeats - beat_time)) <= 0.05
            )
            if i == 0 or i - 1 >= len(raw_bpm):
                row = [i + 1, f"{beat_time:.6f}", int(is_downbeat), "", "", ""]
            else:
                row = [
                    i + 1,
                    f"{beat_time:.6f}",
                    int(is_downbeat),
                    f"{midpoints[i - 1]:.6f}",
                    f"{raw_bpm[i - 1]:.4f}",
                    f"{smooth_bpm[i - 1]:.4f}",
                ]
            writer.writerow(row)


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
) -> None:
    fig, axes = plt.subplots(
        len(analyses),
        1,
        figsize=(13, 3.4 * len(analyses)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, analysis in zip(axes, analyses):
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


def analyse_file(
    audio_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    estimator: BeatThisEstimator,
) -> list[dict]:
    y, sr = librosa.load(audio_path, sr=args.sample_rate, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    stem = safe_stem(audio_path)
    file_output = output_root / stem
    file_output.mkdir(parents=True, exist_ok=True)

    results, frames = estimator(y, sr)
    analyses: list[dict] = []
    for result in results:
        result.beat_times = np.unique(
            result.beat_times[np.isfinite(result.beat_times)]
        )
        midpoints, raw_bpm, smooth_bpm = local_tempo(result.beat_times)
        main_bpm = modal_tempo(midpoints, smooth_bpm, args.min_bpm, args.max_bpm)
        regions = detect_change_regions(
            midpoints,
            smooth_bpm,
            main_bpm,
            relative_threshold=args.change_ratio,
            absolute_threshold=args.change_bpm,
            min_intervals=args.min_change_beats,
            min_duration=args.min_change_seconds,
        )
        analysis = {
            "result": result,
            "midpoints": midpoints,
            "raw_bpm": raw_bpm,
            "smooth_bpm": smooth_bpm,
            "main_bpm": main_bpm,
            "regions": regions,
            "duration": duration,
            "audio": audio_path,
        }
        analyses.append(analysis)
        write_beats_csv(
            file_output / f"{stem}__{result.method}__beats.csv",
            result,
            midpoints,
            raw_bpm,
            smooth_bpm,
        )
        write_click_track(
            file_output / f"{stem}__{result.method}__clicks.wav", y, sr, result
        )

    write_frame_predictions_csv(
        file_output / f"{stem}__beat-this__frames.csv", frames
    )
    write_plot(file_output / f"{stem}__tempo.png", audio_path.name, analyses, duration)
    write_probability_plot(
        file_output / f"{stem}__probabilities.png",
        audio_path.name,
        frames,
        duration,
    )
    report = {
        "audio": str(audio_path.resolve()),
        "duration_seconds": round(duration, 3),
        "frame_inference": {
            "fps": frames.fps,
            "window_seconds": frames.window_seconds,
            "hop_seconds": frames.hop_seconds,
            "overlap_windows": frames.overlap_windows,
            "aggregation": "Hann-weighted logit mean",
        },
        "interpretation_note": (
            "Compare raw and fused click-track WAVs. Highlighted regions are candidates, "
            "not ground truth. "
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
            print(f"Beat This! failed for {audio_path.name}: {exc}", file=sys.stderr)
            return 1

    summary_path = args.output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "audio",
                "method",
                "duration_seconds",
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
