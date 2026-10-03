"""Small Python API for the CCB production pipeline.

Example:

    from API import run, set_click_gain, set_music_gain

    set_music_gain(0.25)
    set_click_gain(0.75)
    result = run("song.mp3")
    print(result.beats_csv)
    print(result.report["result"]["dominant_bpm"])
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import CCB as ccb


API_VERSION = "1.0"
__version__ = ccb.CCB_VERSION


class CCBError(Exception):
    """Base class for errors intentionally exposed by the CCB API."""


class InvalidArgumentError(CCBError, ValueError):
    """A public API argument has an invalid value."""


class ResourceNotFoundError(CCBError, FileNotFoundError):
    """An input audio file or previously generated result is unavailable."""


class ResultStateError(CCBError, ValueError):
    """Stored result files are incomplete, inconsistent, or belong elsewhere."""


class EditConflictError(InvalidArgumentError):
    """A requested manual edit conflicts with an existing beat or range."""


class ItemNotFoundError(CCBError, KeyError):
    """A requested beat or NO_BEAT identifier does not exist."""


_music_gain = ccb.DEFAULT_MUSIC_GAIN
_click_gain = ccb.DEFAULT_CLICK_GAIN


def set_music_gain(value: float) -> None:
    """Set the process-wide music gain used by subsequent ``run`` calls."""
    global _music_gain
    try:
        music_gain, _ = ccb.validate_mix_gains(value, _click_gain)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    _music_gain = music_gain


def set_click_gain(value: float) -> None:
    """Set the process-wide click gain used by subsequent ``run`` calls."""
    global _click_gain
    try:
        _, click_gain = ccb.validate_mix_gains(_music_gain, value)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    _click_gain = click_gain


@dataclass(frozen=True)
class RunResult:
    """Paths and report produced by one successful CCB run."""

    audio: Path
    output_directory: Path
    beats_csv: Path
    click_wav: Path | None
    overview_png: Path
    segments_csv: Path
    report_json: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class NoBeatRange:
    """One time-ordered NO_BEAT range from a song's segments file."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    source: str
    note: str


@dataclass(frozen=True)
class Beat:
    """One beat from the current time-ordered beat grid."""

    beat_id: int
    time_seconds: float
    is_downbeat: bool
    reliability_class: str
    reliability_reason: str


@dataclass(frozen=True)
class CacheEntry:
    """One time-ordered local Beat This! inference cache."""

    cache_directory: Path
    audio_path: Path | None
    song_name: str
    size_bytes: int
    last_used_at: datetime
    status: str


@dataclass(frozen=True)
class CacheCleanupResult:
    """Summary of a cache-pruning operation."""

    deleted: tuple[CacheEntry, ...]
    kept: tuple[CacheEntry, ...]
    freed_bytes: int
    dry_run: bool

    @property
    def deleted_count(self) -> int:
        return len(self.deleted)

    @property
    def kept_count(self) -> int:
        return len(self.kept)


@dataclass(frozen=True)
class ManualBeatAddition:
    time_seconds: float
    is_downbeat: bool
    note: str


@dataclass(frozen=True)
class ManualBeatAdjustment:
    original_time_seconds: float
    new_time_seconds: float
    is_downbeat: bool
    note: str


@dataclass(frozen=True)
class ManualBeatDeletion:
    original_time_seconds: float


@dataclass(frozen=True)
class ManualBeatEdits:
    added: tuple[ManualBeatAddition, ...]
    adjusted: tuple[ManualBeatAdjustment, ...]
    deleted: tuple[ManualBeatDeletion, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ReviewRange:
    start_seconds: float
    end_seconds: float
    classification: str
    reliability_score: float
    reason: str


@dataclass(frozen=True)
class SongInfo:
    audio_path: Path
    output_directory: Path
    duration_seconds: float | None
    result_status: str
    cache_status: str
    beat_count: int
    downbeat_count: int
    dominant_bpm: float | None
    manual_added: int
    manual_adjusted: int
    manual_deleted: int
    no_beat_count: int
    last_processed_at: datetime | None
    last_cache_used_at: datetime | None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def _cache_roots(
    output_dir: str | Path,
    cache_dir: str | Path | None,
) -> list[Path]:
    if cache_dir is not None:
        candidates = [Path(cache_dir).expanduser()]
    elif os.environ.get("CCB_CACHE_DIR"):
        candidates = [ccb.default_cache_root()]
    else:
        candidates = [
            ccb.default_cache_root(),
            ccb.legacy_cache_root(Path(output_dir).expanduser()),
        ]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return roots


def _cache_entry(path: Path) -> CacheEntry:
    required = {
        "metadata": path / "inference.csv",
        "frames": path / "frames.csv",
        "fused": path / "fused_beats.csv",
    }
    audio_path: Path | None = None
    status = "INCOMPLETE"
    if required["metadata"].is_file():
        try:
            metadata = ccb.read_inference_metadata_csv(required["metadata"])
            audio_path = Path(metadata.audio_path).expanduser().resolve()
            if all(item.is_file() for item in required.values()):
                status = "READY" if audio_path.is_file() else "SOURCE_MISSING"
        except (OSError, ValueError, TypeError, KeyError):
            status = "INVALID"
    size_bytes = sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )
    return CacheEntry(
        cache_directory=path.resolve(),
        audio_path=audio_path,
        song_name=(audio_path.stem if audio_path is not None else path.name),
        size_bytes=size_bytes,
        last_used_at=datetime.fromtimestamp(path.stat().st_mtime).astimezone(),
        status=status,
    )


def list_caches(
    *,
    output_dir: str | Path = "results",
    cache_dir: str | Path | None = None,
) -> list[CacheEntry]:
    """List system/custom and compatible legacy caches, newest first."""
    entries: list[CacheEntry] = []
    for root in _cache_roots(output_dir, cache_dir):
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir():
                entries.append(_cache_entry(path))
    return sorted(
        entries,
        key=lambda item: (-item.last_used_at.timestamp(), item.song_name.casefold()),
    )


def prune_caches(
    keep: int | Iterable[str | Path] | None = None,
    *,
    output_dir: str | Path = "results",
    cache_dir: str | Path | None = None,
    dry_run: bool = False,
) -> CacheCleanupResult:
    """Delete all caches except the newest N or caches for selected audio files."""
    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a boolean")
    entries = list_caches(output_dir=output_dir, cache_dir=cache_dir)
    if keep is None:
        kept: list[CacheEntry] = []
    elif isinstance(keep, int) and not isinstance(keep, bool):
        if keep < 0:
            raise InvalidArgumentError("keep must be non-negative")
        kept = entries[:keep]
    else:
        if isinstance(keep, (str, bytes, Path)) or not isinstance(keep, Iterable):
            raise TypeError("keep must be an integer, a file list, or None")
        keep_items = list(keep)
        if any(not isinstance(item, (str, Path)) for item in keep_items):
            raise TypeError("every item in keep must be a string or Path")
        keep_paths = {Path(item).expanduser().resolve() for item in keep_items}
        kept = [item for item in entries if item.audio_path in keep_paths]
    kept_directories = {item.cache_directory for item in kept}
    deleted = [
        item for item in entries if item.cache_directory not in kept_directories
    ]
    if not dry_run:
        allowed_roots = {
            root.resolve() for root in _cache_roots(output_dir, cache_dir)
        }
        for item in deleted:
            target = item.cache_directory.resolve()
            if target.parent not in allowed_roots or target == target.parent:
                raise ResultStateError(
                    f"Refusing to delete unsafe cache path: {target}"
                )
            shutil.rmtree(target)
    return CacheCleanupResult(
        deleted=tuple(deleted),
        kept=tuple(kept),
        freed_bytes=sum(item.size_bytes for item in deleted),
        dry_run=dry_run,
    )


def _result_paths(
    file_name: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    audio_path = Path(file_name).expanduser()
    if not audio_path.is_file():
        raise ResourceNotFoundError(f"Audio file does not exist: {audio_path}")
    result_dir = ccb.result_directory(
        audio_path, Path(output_dir).expanduser()
    )
    return (
        audio_path,
        result_dir / "beats.csv",
        result_dir / "segments.csv",
        result_dir / "report.json",
    )


def _no_beat_paths(
    file_name: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    audio_path, _, segments_path, report_path = _result_paths(file_name, output_dir)
    if not segments_path.is_file() or not report_path.is_file():
        raise ResourceNotFoundError(
            "NO_BEAT settings do not exist yet; run API.run() for this audio first"
        )
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    report_audio = report.get("audio")
    if not report_audio or Path(report_audio).resolve() != audio_path.resolve():
        raise ResultStateError(
            "The existing result belongs to a different audio file with the same name"
        )
    return audio_path, segments_path, report_path


def _load_beat_state(
    file_name: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, str]]]:
    audio_path, beats_path, _, report_path = _result_paths(file_name, output_dir)
    if not beats_path.is_file() or not report_path.is_file():
        raise ResourceNotFoundError(
            "Beat settings do not exist yet; run API.run() for this audio first"
        )
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if Path(report.get("audio", "")).resolve() != audio_path.resolve():
        raise ResultStateError(
            "The existing result belongs to a different audio file with the same name"
        )
    with beats_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["beat_time_seconds"]))
    return beats_path, report_path, report, rows


def _as_beat(row: dict[str, str]) -> Beat:
    return Beat(
        beat_id=int(row["beat_index"]),
        time_seconds=float(row["beat_time_seconds"]),
        is_downbeat=ccb._parse_bool(row.get("is_downbeat", "0")),
        reliability_class=row.get("reliability_class", ""),
        reliability_reason=row.get("reliability_reason", ""),
    )


def _validate_beat_time(value: float, duration: float) -> float:
    try:
        time_seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError("Beat time must be a finite number") from exc
    if not math.isfinite(time_seconds):
        raise InvalidArgumentError("Beat time must be a finite number")
    if not 0.0 <= time_seconds <= duration + 1e-9:
        raise InvalidArgumentError(
            f"Beat time must satisfy 0 <= time <= {duration:.6f}"
        )
    return min(time_seconds, duration)


def _validate_beat_collision(
    time_seconds: float,
    rows: list[dict[str, str]],
    *,
    ignored_row: dict[str, str] | None = None,
) -> None:
    for row in rows:
        if row is ignored_row:
            continue
        existing = float(row["beat_time_seconds"])
        if abs(existing - time_seconds) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE:
            raise EditConflictError(
                f"Beat time overlaps existing beat {row['beat_index']} at {existing:.9f}s"
            )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_beats_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "beat_index", "sample_index", "beat_time_seconds", "is_downbeat",
        "activity_segment_id", "is_no_beat", "interval_midpoint_seconds",
        "raw_local_bpm", "smoothed_local_bpm", "reliability_class",
        "reliability_score", "reliability_reason",
    ]
    rows.sort(key=lambda row: float(row["beat_time_seconds"]))
    sample_rate_candidates: list[float] = []
    for row in rows:
        time_seconds = float(row["beat_time_seconds"])
        if (
            time_seconds > 0
            and row.get("sample_index", "")
            and row.get("reliability_class") != "MANUAL_EDIT"
        ):
            sample_rate_candidates.append(float(row["sample_index"]) / time_seconds)
    sample_rate = (
        round(float(ccb.np.median(sample_rate_candidates)))
        if sample_rate_candidates else 22050.0
    )
    activity_ranges: list[tuple[float, float]] = []
    segments_path = path.parent / "segments.csv"
    if segments_path.is_file():
        with segments_path.open(newline="", encoding="utf-8-sig") as handle:
            segment_rows = list(csv.DictReader(handle))
        duration = max(
            (float(row["end_seconds"]) for row in segment_rows), default=0.0
        )
        activity_ranges = ccb.active_ranges(
            ccb.read_segments_csv(segments_path, duration), duration
        )
    beat_times = ccb.np.asarray(
        [float(row["beat_time_seconds"]) for row in rows], dtype=float
    )
    midpoints, raw_bpm, smooth_bpm = ccb.local_tempo(beat_times)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8-sig", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                time_seconds = float(row["beat_time_seconds"])
                activity_id = next(
                    (
                        range_index
                        for range_index, (start, end) in enumerate(activity_ranges)
                        if start <= time_seconds <= end
                    ),
                    -1,
                )
                row["beat_index"] = str(index + 1)
                row["sample_index"] = str(int(round(time_seconds * sample_rate)))
                row["beat_time_seconds"] = f"{time_seconds:.9f}"
                row["activity_segment_id"] = str(activity_id)
                row["is_no_beat"] = str(int(activity_id < 0))
                previous_activity = (
                    int(rows[index - 1].get("activity_segment_id", -1))
                    if index else -1
                )
                if index == 0 or activity_id < 0 or activity_id != previous_activity:
                    row["interval_midpoint_seconds"] = ""
                    row["raw_local_bpm"] = ""
                    row["smoothed_local_bpm"] = ""
                else:
                    row["interval_midpoint_seconds"] = f"{midpoints[index - 1]:.9f}"
                    row["raw_local_bpm"] = f"{raw_bpm[index - 1]:.4f}"
                    row["smoothed_local_bpm"] = f"{smooth_bpm[index - 1]:.4f}"
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _manual_edits(report: dict[str, Any]) -> dict[str, list[dict]]:
    return ccb.normalize_manual_beat_edits(report.get("manual_edits"))


def _find_edit(items: list[dict], key: str, time_seconds: float) -> dict | None:
    return next(
        (
            item for item in items
            if abs(float(item.get(key, float("inf"))) - time_seconds)
            <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
        ),
        None,
    )


def _load_segments(
    file_name: str | Path,
    output_dir: str | Path,
) -> tuple[Path, list[ccb.ActivitySegment], float]:
    _, segments_path, report_path = _no_beat_paths(file_name, output_dir)
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    with segments_path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    active_ends = [
        float(row["end_seconds"])
        for row in raw_rows
        if not ccb._parse_bool(row.get("no_beat", "0"))
    ]
    duration = (
        max(active_ends)
        if active_ends
        else float(report["duration_seconds"])
    )
    return segments_path, ccb.read_segments_csv(segments_path, duration), duration


def _ordered_no_beat_segments(
    segments: list[ccb.ActivitySegment],
) -> list[ccb.ActivitySegment]:
    ordered = sorted(
        (item for item in segments if item.no_beat),
        key=lambda item: (item.start_seconds, item.end_seconds),
    )
    for segment_id, item in enumerate(ordered, start=1):
        item.segment_id = segment_id
    return ordered


def _as_no_beat_range(item: ccb.ActivitySegment) -> NoBeatRange:
    return NoBeatRange(
        segment_id=item.segment_id,
        start_seconds=item.start_seconds,
        end_seconds=item.end_seconds,
        source=item.source,
        note=item.note,
    )


def _validate_range(
    start_seconds: float,
    end_seconds: float,
    duration: float,
) -> tuple[float, float]:
    try:
        start = float(start_seconds)
        end = float(end_seconds)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(
            "NO_BEAT start and end must be finite numbers"
        ) from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise InvalidArgumentError("NO_BEAT start and end must be finite numbers")
    if not 0.0 <= start < end <= duration + 1e-9:
        raise InvalidArgumentError(
            f"NO_BEAT range must satisfy 0 <= start < end <= {duration:.6f}"
        )
    return start, min(end, duration)


def _validate_no_overlap(
    target: ccb.ActivitySegment,
    no_beat_segments: list[ccb.ActivitySegment],
) -> None:
    for item in no_beat_segments:
        if item is target:
            continue
        if target.start_seconds < item.end_seconds and target.end_seconds > item.start_seconds:
            raise EditConflictError(
                "NO_BEAT range overlaps segment "
                f"{item.segment_id}: {item.start_seconds:.6f}-{item.end_seconds:.6f}"
            )


def _write_segments_atomic(
    path: Path,
    segments: list[ccb.ActivitySegment],
) -> None:
    active = sorted(
        (item for item in segments if not item.no_beat),
        key=lambda item: (item.start_seconds, item.end_seconds),
    )
    for item in active:
        if item.source == "default" and item.start_seconds == 0.0:
            item.segment_id = 0
    blocked = _ordered_no_beat_segments(segments)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
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
            for item in [*active, *blocked]:
                writer.writerow(
                    [
                        item.segment_id,
                        f"{item.start_seconds:.9f}",
                        f"{item.end_seconds:.9f}",
                        int(item.no_beat),
                        item.source,
                        item.note,
                    ]
                )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def list_no_beat_ranges(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
) -> list[NoBeatRange]:
    """List all NO_BEAT ranges in chronological ID order."""
    _, segments, _ = _load_segments(file_name, output_dir)
    return [_as_no_beat_range(item) for item in _ordered_no_beat_segments(segments)]


def create_no_beat_range(
    file_name: str | Path,
    start_seconds: float,
    end_seconds: float,
    *,
    output_dir: str | Path = "results",
    note: str = "",
) -> NoBeatRange:
    """Create a user NO_BEAT range and renumber all ranges by time."""
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    segments_path, segments, duration = _load_segments(file_name, output_dir)
    start, end = _validate_range(start_seconds, end_seconds, duration)
    target = ccb.ActivitySegment(
        segment_id=0,
        start_seconds=start,
        end_seconds=end,
        no_beat=True,
        source="user",
        note=note.strip(),
    )
    no_beat_segments = _ordered_no_beat_segments(segments)
    _validate_no_overlap(target, no_beat_segments)
    segments.append(target)
    _write_segments_atomic(segments_path, segments)
    return _as_no_beat_range(target)


def update_no_beat_range(
    file_name: str | Path,
    segment_id: int,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    note: str | None = None,
    output_dir: str | Path = "results",
) -> NoBeatRange:
    """Update one user range, then return its new chronological ID."""
    if not isinstance(segment_id, int) or isinstance(segment_id, bool):
        raise TypeError("segment_id must be an integer")
    if note is not None and not isinstance(note, str):
        raise TypeError("note must be a string or None")
    segments_path, segments, duration = _load_segments(file_name, output_dir)
    no_beat_segments = _ordered_no_beat_segments(segments)
    target = next(
        (item for item in no_beat_segments if item.segment_id == segment_id),
        None,
    )
    if target is None:
        raise ItemNotFoundError(
            f"NO_BEAT segment does not exist: {segment_id}"
        )
    if target.source != "user":
        raise PermissionError(f"NO_BEAT segment {segment_id} is not user-managed")
    start, end = _validate_range(
        target.start_seconds if start_seconds is None else start_seconds,
        target.end_seconds if end_seconds is None else end_seconds,
        duration,
    )
    target.start_seconds = start
    target.end_seconds = end
    if note is not None:
        target.note = note.strip()
    _validate_no_overlap(target, no_beat_segments)
    _write_segments_atomic(segments_path, segments)
    return _as_no_beat_range(target)


def delete_no_beat_range(
    file_name: str | Path,
    segment_id: int,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Delete one user range and renumber the remaining ranges by time."""
    if not isinstance(segment_id, int) or isinstance(segment_id, bool):
        raise TypeError("segment_id must be an integer")
    segments_path, segments, _ = _load_segments(file_name, output_dir)
    no_beat_segments = _ordered_no_beat_segments(segments)
    target = next(
        (item for item in no_beat_segments if item.segment_id == segment_id),
        None,
    )
    if target is None:
        raise ItemNotFoundError(
            f"NO_BEAT segment does not exist: {segment_id}"
        )
    if target.source != "user":
        raise PermissionError(f"NO_BEAT segment {segment_id} is not user-managed")
    segments.remove(target)
    _write_segments_atomic(segments_path, segments)


def clear_no_beat_ranges(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
) -> int:
    """Delete every user-managed NO_BEAT range and return the count."""
    segments_path, segments, _ = _load_segments(file_name, output_dir)
    kept = [
        item
        for item in segments
        if not (item.no_beat and item.source == "user")
    ]
    deleted = len(segments) - len(kept)
    if deleted:
        _write_segments_atomic(segments_path, kept)
    return deleted


def get_result(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
) -> RunResult:
    """Read one existing result without running CCB or touching its cache."""
    audio_path, beats_path, segments_path, report_path = _result_paths(
        file_name, output_dir
    )
    if not report_path.is_file():
        raise ResourceNotFoundError(
            "Result does not exist yet; run API.run() for this audio first"
        )
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if Path(report.get("audio", "")).resolve() != audio_path.resolve():
        raise ResultStateError(
            "The existing result belongs to a different audio file with the same name"
        )
    result_dir = report_path.parent
    result_section = report.get("result", {})
    overview_path = result_dir / result_section.get("overview_png", "overview.png")
    click_name = result_section.get("click_wav")
    click_path = None if not click_name else result_dir / click_name
    required = [beats_path, segments_path, overview_path]
    if click_path is not None:
        required.append(click_path)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ResourceNotFoundError(
            "Existing result is incomplete; missing: " + ", ".join(missing)
        )
    return RunResult(
        audio=audio_path.resolve(),
        output_directory=result_dir.resolve(),
        beats_csv=beats_path.resolve(),
        click_wav=None if click_path is None else click_path.resolve(),
        overview_png=overview_path.resolve(),
        segments_csv=segments_path.resolve(),
        report_json=report_path.resolve(),
        report=report,
    )


def get_manual_beat_edits(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
) -> ManualBeatEdits:
    """Read persisted manual beat operations and matching warnings."""
    report = get_result(file_name, output_dir=output_dir).report
    edits = _manual_edits(report)
    return ManualBeatEdits(
        added=tuple(
            ManualBeatAddition(
                time_seconds=float(item["time_seconds"]),
                is_downbeat=bool(item.get("is_downbeat", False)),
                note=str(item.get("note", "")),
            )
            for item in edits["added"]
            if "time_seconds" in item
        ),
        adjusted=tuple(
            ManualBeatAdjustment(
                original_time_seconds=float(item["original_time_seconds"]),
                new_time_seconds=float(item["new_time_seconds"]),
                is_downbeat=bool(item.get("is_downbeat", False)),
                note=str(item.get("note", "")),
            )
            for item in edits["adjusted"]
            if "original_time_seconds" in item and "new_time_seconds" in item
        ),
        deleted=tuple(
            ManualBeatDeletion(
                original_time_seconds=float(item["original_time_seconds"])
            )
            for item in edits["deleted"]
            if "original_time_seconds" in item
        ),
        warnings=tuple(str(item) for item in report.get("manual_edit_warnings", [])),
    )


def get_review_ranges(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
    classifications: Iterable[str] | None = None,
) -> list[ReviewRange]:
    """Read the report's recommended human-review ranges."""
    selected: set[str] | None = None
    if classifications is not None:
        if isinstance(classifications, (str, bytes)):
            raise TypeError("classifications must be a collection of strings")
        values = list(classifications)
        if any(not isinstance(item, str) for item in values):
            raise TypeError("every classification must be a string")
        selected = set(values)
    report = get_result(file_name, output_dir=output_dir).report
    values = report.get("reliability", {}).get("recommended_review_ranges", [])
    return [
        ReviewRange(
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            classification=str(item["classification"]),
            reliability_score=float(item["reliability_score"]),
            reason=str(item.get("reason", "")),
        )
        for item in values
        if isinstance(item, dict)
        and (selected is None or item.get("classification") in selected)
    ]


def inspect_song(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
    cache_dir: str | Path | None = None,
) -> SongInfo:
    """Summarize result, cache, manual-edit, and NO_BEAT state."""
    audio_path, beats_path, segments_path, report_path = _result_paths(
        file_name, output_dir
    )
    result_dir = report_path.parent
    report: dict[str, Any] = {}
    result_status = "MISSING"
    if report_path.is_file():
        try:
            with report_path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            result_section = report.get("result", {})
            overview = result_dir / result_section.get("overview_png", "overview.png")
            click_name = result_section.get("click_wav")
            required = [beats_path, segments_path, overview]
            if click_name:
                required.append(result_dir / click_name)
            result_status = (
                "READY"
                if Path(report.get("audio", "")).resolve() == audio_path.resolve()
                and all(path.is_file() for path in required)
                else "INCOMPLETE"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            result_status = "INVALID"
            report = {}
    matching_caches = [
        item
        for item in list_caches(output_dir=output_dir, cache_dir=cache_dir)
        if item.audio_path == audio_path.resolve()
    ]
    cache = matching_caches[0] if matching_caches else None
    edits = _manual_edits(report)
    no_beat_count = 0
    if segments_path.is_file() and report.get("duration_seconds") is not None:
        try:
            no_beat_count = len(
                ccb.no_beat_ranges(
                    ccb.read_segments_csv(
                        segments_path, float(report["duration_seconds"])
                    ),
                    float(report["duration_seconds"]),
                )
            )
        except (OSError, ValueError, TypeError, KeyError):
            pass
    result_section = report.get("result", {})
    return SongInfo(
        audio_path=audio_path.resolve(),
        output_directory=result_dir.resolve(),
        duration_seconds=(
            float(report["duration_seconds"])
            if report.get("duration_seconds") is not None else None
        ),
        result_status=result_status,
        cache_status="MISSING" if cache is None else cache.status,
        beat_count=int(result_section.get("beat_count", 0)),
        downbeat_count=int(result_section.get("downbeat_count", 0)),
        dominant_bpm=(
            float(result_section["dominant_bpm"])
            if result_section.get("dominant_bpm") is not None else None
        ),
        manual_added=len(edits["added"]),
        manual_adjusted=len(edits["adjusted"]),
        manual_deleted=len(edits["deleted"]),
        no_beat_count=no_beat_count,
        last_processed_at=(
            datetime.fromtimestamp(report_path.stat().st_mtime).astimezone()
            if report_path.is_file() else None
        ),
        last_cache_used_at=None if cache is None else cache.last_used_at,
    )


def list_beats(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    reliability_class: str | None = None,
    manual_only: bool = False,
) -> list[Beat]:
    """List current beats, optionally filtered without changing their IDs."""
    if reliability_class is not None and not isinstance(reliability_class, str):
        raise TypeError("reliability_class must be a string or None")
    if not isinstance(manual_only, bool):
        raise TypeError("manual_only must be a boolean")
    _, _, report, rows = _load_beat_state(file_name, output_dir)
    duration = float(report["duration_seconds"])
    start = (
        0.0 if start_seconds is None
        else _validate_beat_time(start_seconds, duration)
    )
    end = (
        duration if end_seconds is None
        else _validate_beat_time(end_seconds, duration)
    )
    if start > end:
        raise InvalidArgumentError("start_seconds must not exceed end_seconds")
    for index, row in enumerate(rows, start=1):
        row["beat_index"] = str(index)
    beats = [_as_beat(row) for row in rows]
    return [
        item
        for item in beats
        if start <= item.time_seconds <= end
        and (
            reliability_class is None
            or item.reliability_class == reliability_class
        )
        and (not manual_only or item.reliability_class == "MANUAL_EDIT")
    ]


def validate_result(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
    cache_dir: str | Path | None = None,
) -> ValidationResult:
    """Validate an existing result without modifying files or cache state."""
    errors: list[str] = []
    warnings: list[str] = []
    audio_path, beats_path, segments_path, report_path = _result_paths(
        file_name, output_dir
    )
    if not report_path.is_file():
        return ValidationResult(False, ("report.json is missing",), ())
    try:
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return ValidationResult(False, (f"report.json is invalid: {exc}",), ())
    if Path(report.get("audio", "")).resolve() != audio_path.resolve():
        errors.append("report.json belongs to a different audio file")
    duration = report.get("duration_seconds")
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        duration_seconds = math.nan
        errors.append("duration_seconds is missing or invalid")
    result_section = report.get("result", {})
    result_dir = report_path.parent
    expected_paths = {
        "beats.csv": beats_path,
        "segments.csv": segments_path,
        "overview.png": result_dir / result_section.get(
            "overview_png", "overview.png"
        ),
    }
    click_name = result_section.get("click_wav")
    if click_name:
        expected_paths["click.wav"] = result_dir / click_name
    for label, path in expected_paths.items():
        if not path.is_file():
            errors.append(f"{label} is missing")

    rows: list[dict[str, str]] = []
    if beats_path.is_file():
        try:
            with beats_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            indices = [int(row["beat_index"]) for row in rows]
            if indices != list(range(1, len(rows) + 1)):
                errors.append("beat IDs are not consecutive")
            times = [float(row["beat_time_seconds"]) for row in rows]
            if any(not math.isfinite(value) for value in times):
                errors.append("beat timestamp is not finite")
            if any(
                right - left <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
                for left, right in zip(times, times[1:])
            ):
                errors.append("beat timestamps overlap or are not strictly increasing")
            if math.isfinite(duration_seconds) and any(
                value < 0 or value > duration_seconds + 1e-9 for value in times
            ):
                errors.append("beat timestamp falls outside the audio duration")
            if result_section.get("beat_count") != len(rows):
                errors.append("report beat_count does not match beats.csv")
            downbeat_count = sum(
                ccb._parse_bool(row.get("is_downbeat", "0")) for row in rows
            )
            if result_section.get("downbeat_count") != downbeat_count:
                errors.append("report downbeat_count does not match beats.csv")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"beats.csv is invalid: {exc}")

    blocked_ranges: list[tuple[float, float]] = []
    if segments_path.is_file() and math.isfinite(duration_seconds):
        try:
            segments = ccb.read_segments_csv(segments_path, duration_seconds)
            blocked_ranges = ccb.no_beat_ranges(segments, duration_seconds)
            ordered = sorted(blocked_ranges)
            if any(
                right_start < left_end
                for (_, left_end), (right_start, _) in zip(ordered, ordered[1:])
            ):
                errors.append("NO_BEAT ranges overlap")
            for row in rows:
                time_seconds = float(row["beat_time_seconds"])
                if any(start <= time_seconds <= end for start, end in blocked_ranges):
                    errors.append(
                        f"beat {row['beat_index']} falls inside a NO_BEAT range"
                    )
                    break
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"segments.csv is invalid: {exc}")

    edits = _manual_edits(report)
    visible_times = [float(row["beat_time_seconds"]) for row in rows]
    manual_times = [
        float(row["beat_time_seconds"])
        for row in rows
        if row.get("reliability_class") == "MANUAL_EDIT"
    ]
    for item in edits["added"]:
        if "time_seconds" not in item:
            errors.append("manual added edit is missing time_seconds")
            continue
        target = float(item["time_seconds"])
        hidden = any(start <= target <= end for start, end in blocked_ranges)
        if not hidden and not any(
            abs(value - target) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
            for value in manual_times
        ):
            errors.append(f"manual added beat is missing at {target:.9f}s")
    for item in edits["adjusted"]:
        if "new_time_seconds" not in item:
            errors.append("manual adjusted edit is missing new_time_seconds")
            continue
        target = float(item["new_time_seconds"])
        hidden = any(start <= target <= end for start, end in blocked_ranges)
        if not hidden and not any(
            abs(value - target) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
            for value in manual_times
        ):
            errors.append(f"manual adjusted beat is missing at {target:.9f}s")
    replacement_times = {
        float(item[key])
        for category, key in (("added", "time_seconds"), ("adjusted", "new_time_seconds"))
        for item in edits[category]
        if key in item
    }
    for item in edits["deleted"]:
        if "original_time_seconds" not in item:
            errors.append("manual deleted edit is missing original_time_seconds")
            continue
        target = float(item["original_time_seconds"])
        intentionally_replaced = any(
            abs(value - target) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
            for value in replacement_times
        )
        if not intentionally_replaced and any(
            abs(value - target) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
            for value in visible_times
        ):
            errors.append(f"manually deleted beat reappeared at {target:.9f}s")
    warnings.extend(str(item) for item in report.get("manual_edit_warnings", []))

    matching_caches = [
        item
        for item in list_caches(output_dir=output_dir, cache_dir=cache_dir)
        if item.audio_path == audio_path.resolve()
    ]
    if not matching_caches:
        warnings.append("inference cache is missing")
    elif matching_caches[0].status != "READY":
        warnings.append(f"inference cache status is {matching_caches[0].status}")
    return ValidationResult(not errors, tuple(errors), tuple(warnings))


def create_beat(
    file_name: str | Path,
    time_seconds: float,
    *,
    is_downbeat: bool = False,
    output_dir: str | Path = "results",
    note: str = "",
) -> Beat:
    """Add one manual beat and renumber all beats chronologically."""
    if not isinstance(is_downbeat, bool):
        raise TypeError("is_downbeat must be a boolean")
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    beats_path, report_path, report, rows = _load_beat_state(file_name, output_dir)
    value = _validate_beat_time(time_seconds, float(report["duration_seconds"]))
    _validate_beat_collision(value, rows)
    edits = _manual_edits(report)
    edits["added"].append(
        {"time_seconds": value, "is_downbeat": is_downbeat, "note": note.strip()}
    )
    target = {name: "" for name in (
        "beat_index", "sample_index", "beat_time_seconds", "is_downbeat",
        "activity_segment_id", "is_no_beat", "interval_midpoint_seconds",
        "raw_local_bpm", "smoothed_local_bpm", "reliability_class",
        "reliability_score", "reliability_reason",
    )}
    target.update(
        beat_time_seconds=f"{value:.9f}", is_downbeat=str(int(is_downbeat)),
        activity_segment_id="0", is_no_beat="0",
        reliability_class="MANUAL_EDIT", reliability_score="1.000000",
        reliability_reason="manual_added",
    )
    rows.append(target)
    report["manual_edits"] = edits
    report.setdefault("result", {})["beat_count"] = len(rows)
    report["result"]["downbeat_count"] = sum(
        ccb._parse_bool(row.get("is_downbeat", "0")) for row in rows
    )
    _write_json_atomic(report_path, report)
    _write_beats_atomic(beats_path, rows)
    return next(
        item for item in list_beats(file_name, output_dir=output_dir)
        if abs(item.time_seconds - value) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
    )


def update_beat(
    file_name: str | Path,
    beat_id: int,
    *,
    time_seconds: float | None = None,
    is_downbeat: bool | None = None,
    output_dir: str | Path = "results",
    note: str | None = None,
) -> Beat:
    """Adjust one beat and return its new chronological ID."""
    if not isinstance(beat_id, int) or isinstance(beat_id, bool):
        raise TypeError("beat_id must be an integer")
    if is_downbeat is not None and not isinstance(is_downbeat, bool):
        raise TypeError("is_downbeat must be a boolean or None")
    if note is not None and not isinstance(note, str):
        raise TypeError("note must be a string or None")
    beats_path, report_path, report, rows = _load_beat_state(file_name, output_dir)
    if not 1 <= beat_id <= len(rows):
        raise ItemNotFoundError(f"Beat does not exist: {beat_id}")
    target = rows[beat_id - 1]
    old_time = float(target["beat_time_seconds"])
    new_time = _validate_beat_time(
        old_time if time_seconds is None else time_seconds,
        float(report["duration_seconds"]),
    )
    _validate_beat_collision(new_time, rows, ignored_row=target)
    new_downbeat = (
        ccb._parse_bool(target.get("is_downbeat", "0"))
        if is_downbeat is None else is_downbeat
    )
    edits = _manual_edits(report)
    added = _find_edit(edits["added"], "time_seconds", old_time)
    adjusted = _find_edit(edits["adjusted"], "new_time_seconds", old_time)
    if added is not None:
        added.update(time_seconds=new_time, is_downbeat=new_downbeat)
        if note is not None:
            added["note"] = note.strip()
        reason = "manual_added"
    elif adjusted is not None:
        adjusted.update(new_time_seconds=new_time, is_downbeat=new_downbeat)
        if note is not None:
            adjusted["note"] = note.strip()
        reason = "manual_adjusted"
    else:
        edits["adjusted"].append(
            {
                "original_time_seconds": old_time,
                "new_time_seconds": new_time,
                "is_downbeat": new_downbeat,
                "note": "" if note is None else note.strip(),
            }
        )
        reason = "manual_adjusted"
    target["beat_time_seconds"] = f"{new_time:.9f}"
    target["is_downbeat"] = str(int(new_downbeat))
    target["reliability_class"] = "MANUAL_EDIT"
    target["reliability_score"] = "1.000000"
    target["reliability_reason"] = reason
    report["manual_edits"] = edits
    report.setdefault("result", {})["downbeat_count"] = sum(
        ccb._parse_bool(row.get("is_downbeat", "0")) for row in rows
    )
    _write_json_atomic(report_path, report)
    _write_beats_atomic(beats_path, rows)
    return next(
        item for item in list_beats(file_name, output_dir=output_dir)
        if abs(item.time_seconds - new_time) <= ccb.MANUAL_BEAT_COLLISION_TOLERANCE
    )


def delete_beat(
    file_name: str | Path,
    beat_id: int,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Delete one beat and remember the deletion across subsequent runs."""
    if not isinstance(beat_id, int) or isinstance(beat_id, bool):
        raise TypeError("beat_id must be an integer")
    beats_path, report_path, report, rows = _load_beat_state(file_name, output_dir)
    if not 1 <= beat_id <= len(rows):
        raise ItemNotFoundError(f"Beat does not exist: {beat_id}")
    target = rows.pop(beat_id - 1)
    old_time = float(target["beat_time_seconds"])
    edits = _manual_edits(report)
    added = _find_edit(edits["added"], "time_seconds", old_time)
    adjusted = _find_edit(edits["adjusted"], "new_time_seconds", old_time)
    if added is not None:
        edits["added"].remove(added)
    else:
        original_time = old_time
        if adjusted is not None:
            original_time = float(adjusted["original_time_seconds"])
            edits["adjusted"].remove(adjusted)
        if _find_edit(edits["deleted"], "original_time_seconds", original_time) is None:
            edits["deleted"].append({"original_time_seconds": original_time})
    report["manual_edits"] = edits
    report.setdefault("result", {})["beat_count"] = len(rows)
    report["result"]["downbeat_count"] = sum(
        ccb._parse_bool(row.get("is_downbeat", "0")) for row in rows
    )
    _write_json_atomic(report_path, report)
    _write_beats_atomic(beats_path, rows)


def reset_beat_edits(
    file_name: str | Path,
    *,
    output_dir: str | Path = "results",
) -> int:
    """Forget all manual beat operations; call run() to restore the automatic grid."""
    _, report_path, report, _ = _load_beat_state(file_name, output_dir)
    edits = _manual_edits(report)
    count = sum(len(items) for items in edits.values())
    report["manual_edits"] = ccb.empty_manual_beat_edits()
    _write_json_atomic(report_path, report)
    return count


def run(
    file_name: str | Path,
    output_dir: str | Path = "results",
    *,
    cache_dir: str | Path | None = None,
    no_click: bool = False,
    refresh_cache: bool = False,
    device: str = "cpu",
    checkpoint: str = "final0",
    hop_seconds: float = 10.0,
    sample_rate: int = 22050,
    min_bpm: float = 40.0,
    max_bpm: float = 240.0,
    change_ratio: float = 0.08,
    change_bpm: float = 8.0,
    min_change_beats: int = 4,
    min_change_seconds: float = 2.0,
    preserve_manual_edits: bool = True,
) -> RunResult:
    """Run CCB for one audio file and return its final artifacts.

    Beat This! is initialized only when the hidden inference cache is absent or
    ``refresh_cache`` is true. Unlike the CLI, this function does not rewrite
    the multi-file ``summary.csv`` because one API call represents one song.
    Manual beat operations are reapplied by default; pass
    ``preserve_manual_edits=False`` to discard them and rebuild the automatic
    grid.
    """
    audio_path = Path(file_name).expanduser()
    if not audio_path.is_file():
        raise ResourceNotFoundError(f"Audio file does not exist: {audio_path}")
    if not 0 < min_bpm < max_bpm:
        raise InvalidArgumentError("Require 0 < min_bpm < max_bpm")
    if not 0 < hop_seconds < 30:
        raise InvalidArgumentError("Require 0 < hop_seconds < 30")
    if sample_rate <= 0:
        raise InvalidArgumentError("sample_rate must be positive")
    if min_change_beats < 1:
        raise InvalidArgumentError("min_change_beats must be at least 1")
    if min_change_seconds < 0:
        raise InvalidArgumentError("min_change_seconds must be non-negative")
    if not isinstance(preserve_manual_edits, bool):
        raise TypeError("preserve_manual_edits must be a boolean")

    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_cache_dir = (
        None if cache_dir is None else Path(cache_dir).expanduser()
    )
    music_gain, click_gain = ccb.validate_mix_gains(_music_gain, _click_gain)
    args = argparse.Namespace(
        audio=[audio_path],
        output=output_root,
        sample_rate=sample_rate,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        change_ratio=change_ratio,
        change_bpm=change_bpm,
        min_change_beats=min_change_beats,
        min_change_seconds=min_change_seconds,
        beat_this_device=device,
        beat_this_checkpoint=checkpoint,
        beat_this_hop_seconds=hop_seconds,
        refresh_cache=refresh_cache,
        no_click=no_click,
        music_gain=music_gain,
        click_gain=click_gain,
        preserve_manual_edits=preserve_manual_edits,
        cache_dir=resolved_cache_dir,
    )

    estimator: ccb.BeatThisEstimator | None = None
    if refresh_cache or not ccb.inference_cache_available(
        audio_path,
        output_root,
        resolved_cache_dir,
        sample_rate=sample_rate,
        checkpoint=checkpoint,
        hop_seconds=hop_seconds,
    ):
        estimator = ccb.BeatThisEstimator(
            device,
            checkpoint,
            hop_seconds=hop_seconds,
        )
    ccb.analyse_file(audio_path, output_root, args, estimator)

    result_dir = ccb.result_directory(audio_path, output_root)
    report_path = result_dir / "report.json"
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    return RunResult(
        audio=audio_path.resolve(),
        output_directory=result_dir.resolve(),
        beats_csv=(result_dir / "beats.csv").resolve(),
        click_wav=(None if no_click else (result_dir / "click.wav").resolve()),
        overview_png=(result_dir / "overview.png").resolve(),
        segments_csv=(result_dir / "segments.csv").resolve(),
        report_json=report_path.resolve(),
        report=report,
    )


__all__ = [
    "API_VERSION",
    "__version__",
    "CCBError",
    "InvalidArgumentError",
    "ResourceNotFoundError",
    "ResultStateError",
    "EditConflictError",
    "ItemNotFoundError",
    "RunResult",
    "NoBeatRange",
    "Beat",
    "CacheEntry",
    "CacheCleanupResult",
    "ManualBeatAddition",
    "ManualBeatAdjustment",
    "ManualBeatDeletion",
    "ManualBeatEdits",
    "ReviewRange",
    "SongInfo",
    "ValidationResult",
    "run",
    "set_music_gain",
    "set_click_gain",
    "list_no_beat_ranges",
    "create_no_beat_range",
    "update_no_beat_range",
    "delete_no_beat_range",
    "clear_no_beat_ranges",
    "list_beats",
    "get_result",
    "inspect_song",
    "get_manual_beat_edits",
    "get_review_ranges",
    "validate_result",
    "create_beat",
    "update_beat",
    "delete_beat",
    "reset_beat_edits",
    "list_caches",
    "prune_caches",
]
