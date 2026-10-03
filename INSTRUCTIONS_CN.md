# CCB 使用说明

[English](INSTRUCTIONS.md) | **简体中文** | [项目首页](README_CN.md)

CCB（Context Calibrated Beats）是一个离线音乐节拍网格工具。它使用 Beat This! 生成拍点和 downbeat，再将速度归一化到 `[120, 240)` BPM。

软件只保留一套正式结果，不生成历史阶段文件。

## 环境

支持 Python 3.10–3.12，建议使用 Python 3.12 和虚拟环境。

Windows：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install context-calibrated-beats
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install context-calibrated-beats
```

如需安装当前源码目录，请将最后一条命令中的包名替换为 `-e .`。

## 基本用法

处理一首歌：

```powershell
ccb "D:\Music\song.mp3"
```

同时处理多首歌：

```powershell
ccb "D:\Music\song1.mp3" "D:\Music\song2.wav"
```

指定输出目录：

```powershell
ccb "D:\Music\song.mp3" -o results
```

使用 GPU：

```powershell
ccb "D:\Music\song.mp3" --beat-this-device cuda
```

不生成 click 音轨：

```powershell
ccb "D:\Music\song.mp3" --no-click
```

强制重新运行 Beat This! 推理：

```powershell
ccb "D:\Music\song.mp3" --refresh-cache
```

查看全部参数：

```powershell
ccb --help
```

在源码目录中仍可使用兼容入口 `python CCB.py ...`。

## Python 函数接口

其他 Python 程序可直接调用 `API.run()`，无需构造 CLI 命令：

```python
from context_calibrated_beats import run, set_click_gain, set_music_gain

set_music_gain(0.25)
set_click_gain(0.75)
result = run(r"D:\Music\song.mp3")
print(result.beats_csv)
print(result.click_wav)
print(result.report["result"]["dominant_bpm"])
```

常用参数：

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

`run()` 每次处理一首歌，返回 `RunResult`。其中包含 `beats_csv`、`click_wav`、`overview_png`、`segments_csv`、`report_json` 的绝对路径，以及已解析的 `report` 字典。调用失败时会抛出异常，便于上层程序捕获和处理。单首 API 调用不会重写多歌汇总用的 `summary.csv`。

`set_music_gain()` 和 `set_click_gain()` 设置后续 `run()` 调用的进程级默认值；未设置时使用 `0.1 / 0.9`。两个增益必须是非负有限数，且不能同时为零。它们只影响 `click.wav`，不会使 Beat This! 缓存失效。

### 缓存管理函数

```python
from context_calibrated_beats import list_caches, prune_caches

# 按最近使用时间从新到旧查询。
caches = list_caches()

# 预览：保留最近 10 个，其余将被删除。
preview = prune_caches(keep=10, dry_run=True)

# 正式清理。
result = prune_caches(keep=10)
print(result.deleted_count, result.freed_bytes)

# 仅保留指定音频的缓存。
prune_caches(keep=[r"D:\Music\song1.mp3", r"D:\Music\song2.wav"])

# 删除全部缓存；下次 run() 将重新进行模型推理。
prune_caches()
```

`list_caches()` 返回 `CacheEntry`，包含缓存目录、原音频路径、大小、最近使用时间及 `READY / INCOMPLETE / SOURCE_MISSING / INVALID` 状态。`prune_caches()` 只删除已确认位于缓存根目录下的直接子目录；建议大规模清理前先使用 `dry_run=True`。

### 只读查询函数

这些函数只读取已有结果，不运行模型、不改写输出，也不更新缓存使用时间：

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

# 保留 beats.csv 中的原始 beat_id，只筛选 90–110 秒的手动拍点。
beats = list_beats(
    r"D:\Music\song.mp3",
    start_seconds=90.0,
    end_seconds=110.0,
    manual_only=True,
)
```

`list_beats()` 还支持 `reliability_class="MANUAL_EDIT"` 等可靠性分类筛选。`validate_result()` 返回错误与警告；缺少缓存只产生警告，因为现有正式结果在没有缓存时仍可正常读取。

### 手动节拍函数

首次 `run()` 后，可以直接管理当前的 `beats.csv`：

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

# 使用缓存重新生成 click、总览图和报告，同时保留上述手动编辑。
result = run(r"D:\Music\song.mp3", preserve_manual_edits=True)
```

新增或调整后的时间不能与已有节拍重合，否则函数会抛出 `EditConflictError`（同时也是 `ValueError` 的子类）。每次变更后，所有节拍按时间重新编号为 `1..N`；新增和调整的行在 `beats.csv` 中标为 `MANUAL_EDIT`。删除记录保存在 `report.json`，因此再次运行时不会被自动网格恢复。

`preserve_manual_edits=True` 是默认行为。设为 `False` 会丢弃全部手动操作并输出纯自动网格。`reset_beat_edits()` 只清除操作记录；随后调用一次 `run()` 才会恢复自动网格及相关最终文件。

### NO_BEAT 函数

需要先对音频成功执行至少一次 `run()`，然后可管理其 `NO_BEAT` 区间：

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

每次创建、调整或删除后，所有 `NO_BEAT` 区间都会按开始时间重新编号为 `1..N`，因此 ID 是当前时间顺序号，不是永久标识。调整函数会返回该区间的新 ID。

这些函数只会原子更新 `segments.csv`，不会运行模型或自动重建结果。修改完成后再调用一次 `run()`，即可复用现有缓存更新节拍、click 和报告。

## 输出

默认输出到 `results/`：

```text
results/
  summary.csv
  歌曲名-路径短哈希/
    beats.csv
    click.wav
    overview.png
    segments.csv
    report.json
```

- `beats.csv`：唯一正式拍点结果，包含时间、downbeat、局部 BPM 和可靠性。
- `click.wav`：原音乐与最终 click 混音；使用 `--no-click` 时不生成。
- `overview.png`：拍点证据、最终网格、BPM 和可靠性的综合图。
- `segments.csv`：用户可编辑的 `NO_BEAT` 区间。
- `report.json`：整体 BPM、拍点数、可靠性摘要和建议检查区间。
- `summary.csv`：多首歌的汇总表。

新结果目录使用“可读文件名 + 8 位绝对路径哈希”，例如 `song-4d072156/`。因此不同目录中的同名音频可以共存。已经存在且元数据匹配的旧式 `results/song/` 会继续原地复用，不会自动迁移或丢失手动编辑。

## 隐藏推理缓存

新安装默认把 Beat This! 推理结果保存在操作系统的用户缓存目录中。这不是调试输出，而是重建最终结果所需的内部缓存。缓存位置优先级为：`run(cache_dir=...)` 或 CLI `--cache-dir`、环境变量 `CCB_CACHE_DIR`、系统用户缓存目录。

新缓存目录同样使用“文件名 + 8 位绝对路径哈希”，防止仓库外不同位置的同名音频共享缓存。

旧版本已经生成在输出目录旁 `.ccb-cache/` 中的缓存会继续被自动发现和复用，查询与清理函数也能看到它们，不会因升级而重新推理。

修改 `segments.csv` 后重新执行同一条命令，CCB 会复用缓存，无需重新运行模型。只有更换音频、更改推理设置或显式使用 `--refresh-cache` 时才应刷新。

## 标注 NO_BEAT

首次运行会自动创建 `segments.csv`：

```csv
segment_id,start_seconds,end_seconds,no_beat,source,note
0,0.000000000,247.440000000,0,default,
```

如需排除 32.5–40.0 秒，可增加：

```csv
1,32.500000000,40.000000000,1,user,spoken intro
```

`no_beat=1` 的区间不参与最终网格、BPM 统计和 click 合成。重新执行命令即可从现有缓存快速重建结果。

## 可靠性

CCB 使用以下标签：

- `RELIABLE`
- `PHASE_REPAIRED`
- `TEMPO_MOTION`
- `NO_BEAT`
- `BEAT_THIS_UNRELIABLE`

标签只用于提示和复核，不会另外改写最终拍点。
