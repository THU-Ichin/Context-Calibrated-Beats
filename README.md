# Context Calibrated Beats (CCB)

*Automatic Beat Tracking, Calibration and Click Generation*

**English** | [简体中文](README_CN.md)

CCB is an offline beat-grid tool. It uses
[Beat This!](https://github.com/CPJKU/beat_this) to detect beats and downbeats,
then builds a continuous, audible, and reviewable beat grid whose final tempo
is normalized to the `[120, 240)` BPM range.

CCB exposes one final result instead of a collection of intermediate pipeline
artifacts. It can be used from the command line or through the
`context_calibrated_beats` Python API.

## Features

- Generate a final `beats.csv`, click track, overview image, segment file, and
  JSON report.
- Reuse local inference caches automatically and invalidate them when the audio
  or inference settings change.
- Manage `NO_BEAT` ranges with automatic chronological ID reassignment.
- Create, move, and delete beats while preserving manual edits and preventing
  time collisions.
- Inspect results, reliability labels, manual edits, and cache status.
- Run on CPU or other devices supported by Beat This!, including CUDA.

## Installation

CCB supports Python 3.10–3.12. A virtual environment is recommended.

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install context-calibrated-beats
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install context-calibrated-beats
```

To install the current source checkout instead, replace the package name with
`-e .` in the final command.

## CLI

After installation, use the `ccb` command:

```powershell
ccb "D:\Music\song.mp3"
ccb "D:\Music\song.mp3" -o results --beat-this-device cuda
ccb "song1.mp3" "song2.wav" --no-click
ccb --help
ccb --version
```

You can also run the source entry point directly:

```powershell
python CCB.py "D:\Music\song.mp3"
```

Results are written to `results/` in the current directory by default. Input
audio may live outside the repository. CCB uses a short hash of the absolute
audio path to distinguish files with the same name in different directories.

## Python API

```python
from context_calibrated_beats import run, set_click_gain, set_music_gain

set_music_gain(0.1)
set_click_gain(0.9)

result = run(r"D:\Music\song.mp3")
print(result.beats_csv)
print(result.click_wav)
print(result.report["result"]["dominant_bpm"])
```

Each `run()` call processes one song. By default, it reuses a valid cache and
preserves manual beat edits. Set `refresh_cache=True` to force fresh inference,
or `preserve_manual_edits=False` to rebuild a fully automatic grid.

### Public functions

| Category | Functions |
| --- | --- |
| Processing | `run()` |
| Mix | `set_music_gain()`, `set_click_gain()` |
| Beats | `list_beats()`, `create_beat()`, `update_beat()`, `delete_beat()`, `reset_beat_edits()` |
| NO_BEAT | `list_no_beat_ranges()`, `create_no_beat_range()`, `update_no_beat_range()`, `delete_no_beat_range()`, `clear_no_beat_ranges()` |
| Read-only inspection | `get_result()`, `inspect_song()`, `get_manual_beat_edits()`, `get_review_ranges()`, `validate_result()` |
| Cache | `list_caches()`, `prune_caches()` |

Callers can catch the public `CCBError` base class. More specific exceptions
include `InvalidArgumentError`, `ResourceNotFoundError`, `ResultStateError`,
`EditConflictError`, and `ItemNotFoundError`.

See the [full English instructions](INSTRUCTIONS.md) for more examples.

## Output layout

```text
results/
  summary.csv
  song-name-path-hash/
    beats.csv
    click.wav
    overview.png
    segments.csv
    report.json
```

- `beats.csv`: the single authoritative beat result, including downbeats,
  local BPM, and reliability.
- `click.wav`: a mix of the source audio and final beat clicks.
- `overview.png`: an overview of beat evidence, the final grid, BPM, and
  reliability.
- `segments.csv`: editable `NO_BEAT` ranges.
- `report.json`: result summary, manual edits, and suggested review ranges.

## Cache and privacy

Inference caches stay on the user's own computer and default to the operating
system's per-user cache directory. Override the location with
`run(cache_dir=...)`, CLI `--cache-dir`, or the `CCB_CACHE_DIR` environment
variable. Caches are not uploaded to this repository or any other service.

New caches record the audio size and modification time, inference settings,
Beat This! version, and CCB cache schema version. Compatible legacy
`.ccb-cache/` directories remain readable when their metadata matches.

## Development and tests

```powershell
python -m unittest discover -s tests
```

## License

CCB is released by Ichin under the [MIT License](LICENSE). Beat This! and other
dependencies retain their respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Users are responsible for
having the rights required to process their input audio.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).
