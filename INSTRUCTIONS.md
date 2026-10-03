# CCB Instructions

**English** | [简体中文](INSTRUCTIONS_CN.md) | [Project home](README.md)

CCB (Context Calibrated Beats) is an offline beat-grid tool. It uses Beat This!
to detect beats and downbeats, then normalizes the tempo to the `[120, 240)` BPM
range.

The application keeps one final result and does not emit historical `raw`,
`fused`, or `repaired` stage files.

## Environment

CCB is available from
[PyPI](https://pypi.org/project/context-calibrated-beats/) and supports Python
3.10–3.12. Python 3.12 and a virtual environment are recommended.

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

Confirm that the command-line entry point is available:

```console
ccb --version
```

Upgrade an existing installation with
`python -m pip install --upgrade context-calibrated-beats`. To install a source
checkout for development, run `python -m pip install -e .` from the repository
root.

## Basic usage

Process one song:

```powershell
ccb "D:\Music\song.mp3"
```

Process multiple songs:

```powershell
ccb "D:\Music\song1.mp3" "D:\Music\song2.wav"
```

Choose an output directory:

```powershell
ccb "D:\Music\song.mp3" -o results
```

Use a GPU:

```powershell
ccb "D:\Music\song.mp3" --beat-this-device cuda
```

Skip click-track generation:

```powershell
ccb "D:\Music\song.mp3" --no-click
```

Force fresh Beat This! inference:

```powershell
ccb "D:\Music\song.mp3" --refresh-cache
```

Show all options:

```powershell
ccb --help
```

From a source checkout, `python CCB.py ...` remains available as a compatible
entry point.

## Python function API

Other Python programs can call the public function API directly instead of
constructing a CLI command:

```python
from context_calibrated_beats import run, set_click_gain, set_music_gain

set_music_gain(0.25)
set_click_gain(0.75)
result = run(r"D:\Music\song.mp3")
print(result.beats_csv)
print(result.click_wav)
print(result.report["result"]["dominant_bpm"])
```

Common parameters:

```python
result = run(
    r"D:\Music\song.mp3",
    output_dir="results",
    cache_dir=None,
    no_click=False,
    refresh_cache=False,
    device="cpu",
    preserve_manual_edits=True,
)
```

Each `run()` call processes one song and returns a `RunResult`. It contains the
absolute paths `beats_csv`, `click_wav`, `overview_png`, `segments_csv`, and
`report_json`, together with the parsed `report` dictionary. Failures raise
exceptions so the calling application can handle them. A single-song API call
does not rewrite the multi-song `summary.csv`.

`set_music_gain()` and `set_click_gain()` set process-wide defaults for later
`run()` calls. The defaults are `0.1 / 0.9`. Both gains must be finite and
non-negative, and they cannot both be zero. They affect only `click.wav` and do
not invalidate the Beat This! inference cache.

### Cache management

```python
from context_calibrated_beats import list_caches, prune_caches

# List caches from most recently used to oldest.
caches = list_caches()

# Preview deletion while retaining the 10 newest caches.
preview = prune_caches(keep=10, dry_run=True)

# Perform the cleanup.
result = prune_caches(keep=10)
print(result.deleted_count, result.freed_bytes)

# Retain caches only for selected audio files.
prune_caches(keep=[r"D:\Music\song1.mp3", r"D:\Music\song2.wav"])

# Delete every cache. The next run() will perform model inference again.
prune_caches()
```

`list_caches()` returns `CacheEntry` objects containing the cache directory,
source audio path, size, last-used time, and a `READY`, `INCOMPLETE`,
`SOURCE_MISSING`, or `INVALID` status. `prune_caches()` deletes only verified
direct children of a cache root. Use `dry_run=True` before a large cleanup.

### Read-only inspection

These functions read existing results without running the model, rewriting
outputs, or updating the cache's last-used time:

```python
from context_calibrated_beats import (
    get_manual_beat_edits,
    get_result,
    get_review_ranges,
    inspect_song,
    list_beats,
    validate_result,
)

result = get_result(r"D:\Music\song.mp3")
info = inspect_song(r"D:\Music\song.mp3")
edits = get_manual_beat_edits(r"D:\Music\song.mp3")
review_ranges = get_review_ranges(r"D:\Music\song.mp3")
validation = validate_result(r"D:\Music\song.mp3")

# Preserve beat_id values from beats.csv while selecting manual beats at 90–110 s.
beats = list_beats(
    r"D:\Music\song.mp3",
    start_seconds=90.0,
    end_seconds=110.0,
    manual_only=True,
)
```

`list_beats()` also supports reliability filters such as
`reliability_class="MANUAL_EDIT"`. `validate_result()` returns errors and
warnings. A missing cache is only a warning because an existing final result
remains readable without its cache.

### Manual beat editing

After the first successful `run()`, the current `beats.csv` can be managed
directly:

```python
from context_calibrated_beats import (
    create_beat,
    delete_beat,
    list_beats,
    run,
    update_beat,
)

created = create_beat(r"D:\Music\song.mp3", 12.345, is_downbeat=False)
updated = update_beat(
    r"D:\Music\song.mp3",
    created.beat_id,
    time_seconds=12.400,
    is_downbeat=True,
)
beats = list_beats(r"D:\Music\song.mp3")
delete_beat(r"D:\Music\song.mp3", updated.beat_id)

# Rebuild the click track, overview, and report from cache while keeping edits.
result = run(r"D:\Music\song.mp3", preserve_manual_edits=True)
```

A new or moved beat cannot overlap an existing beat. A conflict raises
`EditConflictError`, which is also a subclass of `ValueError`. After every
change, all beats are renumbered chronologically from `1..N`. New and adjusted
rows are marked `MANUAL_EDIT` in `beats.csv`. Deletions are recorded in
`report.json`, so another run does not restore them from the automatic grid.

`preserve_manual_edits=True` is the default. Setting it to `False` discards all
manual operations and emits a fully automatic grid. `reset_beat_edits()` only
clears the operation history; call `run()` afterward to restore the automatic
grid and its related final files.

### NO_BEAT ranges

Run the audio successfully at least once before managing its `NO_BEAT` ranges:

```python
from context_calibrated_beats import (
    clear_no_beat_ranges,
    create_no_beat_range,
    delete_no_beat_range,
    list_no_beat_ranges,
    run,
    update_no_beat_range,
)

created = create_no_beat_range(
    r"D:\Music\song.mp3",
    32.5,
    40.0,
    note="spoken section",
)

updated = update_no_beat_range(
    r"D:\Music\song.mp3",
    created.segment_id,
    end_seconds=41.0,
)

ranges = list_no_beat_ranges(r"D:\Music\song.mp3")
delete_no_beat_range(r"D:\Music\song.mp3", updated.segment_id)
deleted_count = clear_no_beat_ranges(r"D:\Music\song.mp3")
```

After every create, update, or delete operation, all `NO_BEAT` ranges are
renumbered chronologically from `1..N`. An ID is therefore the current time
order, not a permanent identifier. The update function returns the range's new
ID.

These functions update `segments.csv` atomically. They do not run the model or
rebuild outputs automatically. Call `run()` after editing to reuse the existing
cache and update the beat grid, click track, and report.

## Output

The default output root is `results/`:

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

- `beats.csv`: the single authoritative beat result, including time, downbeat,
  local BPM, and reliability.
- `click.wav`: a mix of the source audio and final beat clicks; omitted with
  `--no-click`.
- `overview.png`: a combined view of beat evidence, the final grid, BPM, and
  reliability.
- `segments.csv`: user-editable `NO_BEAT` ranges.
- `report.json`: overall BPM, beat count, reliability summary, and suggested
  review ranges.
- `summary.csv`: a summary for multi-song CLI runs.

New result directories use a readable file name plus an eight-character hash
of the absolute path, for example `song-4d072156/`. Files with the same name in
different directories can therefore coexist. A matching legacy directory such
as `results/song/` is reused in place, without migration or loss of manual
edits.

## Hidden inference cache

New installations store Beat This! inference results in the operating system's
per-user cache directory. This is not debug output: it allows CCB to rebuild a
final result without rerunning the model. Cache-location priority is an
explicit `run(cache_dir=...)` or CLI `--cache-dir`, the `CCB_CACHE_DIR`
environment variable, and finally the system user cache directory.

New cache directories also use a file name plus an eight-character absolute
path hash, preventing same-named audio outside the repository from sharing a
cache.

Legacy `.ccb-cache/` data beside an output directory remains discoverable and
reusable. Cache listing and cleanup functions include it, so an upgrade does
not force fresh inference.

After editing `segments.csv`, rerun the same command to rebuild from cache.
Fresh inference is needed only when the audio or inference settings change, or
when `--refresh-cache` is explicitly supplied.

## Marking NO_BEAT ranges

The first run creates `segments.csv` automatically:

```csv
segment_id,start_seconds,end_seconds,no_beat,source,note
0,0.000000000,247.440000000,0,default,
```

To exclude 32.5–40.0 seconds, add:

```csv
1,32.500000000,40.000000000,1,user,spoken intro
```

A range with `no_beat=1` is excluded from the final grid, BPM statistics, and
click synthesis. Rerun the command to rebuild quickly from the existing cache.

## Reliability

CCB uses the following labels:

- `RELIABLE`
- `PHASE_REPAIRED`
- `TEMPO_MOTION`
- `NO_BEAT`
- `BEAT_THIS_UNRELIABLE`

These labels are for review and diagnostics only; they do not independently
rewrite the final beat positions.
