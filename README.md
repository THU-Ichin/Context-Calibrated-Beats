# Context Calibrated Beats (CCB)

*Automatic Beat Tracking, Calibration and Click Generation*

CCB 是一个离线音乐节拍网格工具。它使用
[Beat This!](https://github.com/CPJKU/beat_this) 生成 beat/downbeat，再构建连续、
可试听、可检查的节拍网格，并将最终速度归一化到 `[120, 240)` BPM。

CCB 只保留一套最终结果，不向用户输出各个历史处理阶段。它既可以作为命令行工具
使用，也可以通过 `API.py` 中的函数调用。

## 功能

- 生成最终 `beats.csv`、click 音轨、总览图、区间文件和 JSON 报告。
- 自动复用本机推理缓存；音频或模型设置变化时自动判定缓存失效。
- 管理 `NO_BEAT` 区间，并按时间顺序自动重新编号。
- 新增、调整、删除节拍，保留人工修改并防止时间重合。
- 查询结果、可靠性、人工编辑和缓存状态。
- 支持 CPU，以及 Beat This! 支持的 CUDA 等设备。

## 安装

需要 Python 3.10–3.12。建议使用虚拟环境：

```powershell
git clone https://github.com/THU-Ichin/Context-Calibrated-Beats.git
cd Context-Calibrated-Beats
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux/macOS 请将最后两条命令中的 Python 路径换为 `.venv/bin/python`。

## CLI

安装后可使用 `ccb`：

```powershell
ccb "D:\Music\song.mp3"
ccb "D:\Music\song.mp3" -o results --beat-this-device cuda
ccb "song1.mp3" "song2.wav" --no-click
ccb --help
ccb --version
```

也可以在源码目录直接运行：

```powershell
python CCB.py "D:\Music\song.mp3"
```

默认结果写入当前目录的 `results/`。输入音频可以位于仓库之外；CCB 使用音频绝对
路径的短哈希区分不同目录中的同名文件。

## Python API

```python
from API import run, set_click_gain, set_music_gain

set_music_gain(0.1)
set_click_gain(0.9)

result = run(r"D:\Music\song.mp3")
print(result.beats_csv)
print(result.click_wav)
print(result.report["result"]["dominant_bpm"])
```

`run()` 每次处理一首歌。默认复用有效缓存并保留人工节拍编辑；使用
`refresh_cache=True` 可强制重新推理，使用 `preserve_manual_edits=False` 可输出纯
自动网格。

### 公开函数

| 类别 | 函数 |
| --- | --- |
| 处理 | `run()` |
| 混音 | `set_music_gain()`, `set_click_gain()` |
| 节拍 | `list_beats()`, `create_beat()`, `update_beat()`, `delete_beat()`, `reset_beat_edits()` |
| NO_BEAT | `list_no_beat_ranges()`, `create_no_beat_range()`, `update_no_beat_range()`, `delete_no_beat_range()`, `clear_no_beat_ranges()` |
| 只读查询 | `get_result()`, `inspect_song()`, `get_manual_beat_edits()`, `get_review_ranges()`, `validate_result()` |
| 缓存 | `list_caches()`, `prune_caches()` |

调用方可以捕获公开异常基类 `CCBError`。更具体的类型包括
`InvalidArgumentError`、`ResourceNotFoundError`、`ResultStateError`、
`EditConflictError` 和 `ItemNotFoundError`。

更完整的调用示例见 [中文使用说明](BPM测试说明.md)。

## 输出结构

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

- `beats.csv`：唯一正式拍点结果，包含 downbeat、局部 BPM 和可靠性。
- `click.wav`：原音乐与节拍 click 的混音。
- `overview.png`：节拍证据、最终网格、BPM 和可靠性总览。
- `segments.csv`：可管理的 `NO_BEAT` 区间。
- `report.json`：结果摘要、人工编辑及建议检查区间。

## 缓存与隐私

推理缓存在用户自己的电脑上，默认位于操作系统的用户缓存目录。可通过
`run(cache_dir=...)`、CLI `--cache-dir` 或环境变量 `CCB_CACHE_DIR` 更改位置。
缓存不会上传到本仓库或其他服务。

新缓存会记录音频大小、修改时间、模型设置、Beat This! 版本和 CCB 缓存格式版本。
旧版 `.ccb-cache/` 在元数据匹配时仍可兼容读取。

## 开发与测试

```powershell
python -m unittest discover -s tests
```

## 许可

CCB 由 Ichin 以 [MIT License](LICENSE) 发布。Beat This! 及其他依赖保留各自许可；
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。用户应确保自己有权处理输入
音频。
