# Beat This! BPM/节拍分析

脚本 `compare_bpm.py` 使用 `Beat This!` 预训练 Transformer 输出 beat 和 downbeat，
并生成局部 BPM、疑似变速区间和试听用 click-track。脚本同时保留官方分块基线，
并使用重叠窗口融合逐帧 logits，以减少约 30 秒分块处的硬接缝。
融合结果随后进入五层网格解码器，在 `0.25x / 0.5x / 1x / 2x / 4x`
候选中选择路径，并将最终 click 网格归一化到 `[120, 240)` BPM。
推理产生的基础数据会先保存到 CSV，再从 CSV 重载并生成归一化结果、统计、
click-track 和图片，避免落盘数据与绘图数据来自不同处理阶段。

`librosa` 只用于音频读取、时长计算、click 合成和归一化，不再参与节拍检测。

## 安装

建议使用 Python 3.10～3.12，并建立独立虚拟环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-bpm.txt
```

`Beat This!` 第一次运行时可能需要联网下载预训练权重。脚本不再提供其他节拍检测器作为回退；若模型无法初始化或推理失败，会返回错误并停止。

## 运行

分析一首歌：

```powershell
python compare_bpm.py "audio\Reply.mp3" -o bpm_results
```

一次分析多首：

```powershell
python compare_bpm.py "D:\Music\song1.mp3" "D:\Music\song2.wav" -o bpm_results
```

有 NVIDIA GPU 时可以尝试：

```powershell
python compare_bpm.py "D:\Music\song.mp3" -o bpm_results --beat-this-device cuda
```

可以限制主导 BPM 统计和变速分析所采用的速度范围：

```powershell
python compare_bpm.py "D:\Music\song.mp3" -o bpm_results --min-bpm 70 --max-bpm 190
```

该范围不限制 Beat This! 模型本身的推理。

重叠推理固定使用 30 秒窗口，默认每 10 秒启动一个窗口。可以调整 hop；数值越小，
同一时刻参与融合的窗口越多，但推理耗时也越长：

```powershell
python compare_bpm.py "audio\Reply.mp3" -o bpm_results --beat-this-hop-seconds 5
```

首次运行后，每首歌目录会生成可编辑的 `__segments.csv`。修改其中的
`NO_BEAT` 区段后，可以复用已经保存的 Beat This! 推理结果快速重建派生文件：

```powershell
python compare_bpm.py "audio\Reply.mp3" -o bpm_results --reuse-inference
```

该模式不会重新运行模型，但会重新生成 normalized CSV、三套 click、图片、
报告和汇总。音频路径、输出目录和歌曲文件名必须与首次运行一致。

## 标注 NO_BEAT

`__segments.csv` 默认包含一行覆盖全曲的 `no_beat=0`。用户可以追加
`no_beat=1` 的区段；这些行优先于默认行，例如：

```csv
segment_id,start_seconds,end_seconds,no_beat,source,note
0,0.000000000,238.957000000,0,default,
1,0.000000000,17.600000000,1,user,清唱前奏
2,208.600000000,238.957000000,1,user,自由速度尾奏
```

区间采用 `[start_seconds, end_seconds)` 语义。重处理后，`NO_BEAT` 区段：

- 不输出 click；
- 不参与主 BPM 和变速统计；
- 不运行五档网格解码；
- 不会跨越区段补拍；
- 在图片中以灰色阴影显示。

因此 `[120, 240)` 约束只适用于每个有效节拍区段内部，不适用于跨越
`NO_BEAT` 空白的时间差。

## 输出怎么看

每首歌有一个独立子目录，其中包含：

- `__beat-this-raw__beats.csv`：官方 30 秒分块、`keep_first` 拼接后的拍点；
- `__beat-this-fused__beats.csv`：重叠窗口 Hann 加权融合后的拍点；
- `__beat-this-normalized__beats.csv`：五层网格解码和最终间距约束后的拍点；
- 对应的 `__clicks.wav`：raw/fused/normalized 三套试听文件，高音点击代表 downbeat；
- `__beat-this__frames.csv`：50 FPS 的 raw/fused beat、downbeat logits 和概率；
- `__beat-this__grid.csv`：每个区间的基础 BPM、选中倍率、归一化 BPM 和置信度；
- `__inference.csv`：FPS、窗口、hop、采样率和音频路径等推理元数据；
- `__segments.csv`：用户或模型提供的有效节拍/`NO_BEAT` 时间段；
- `__tempo.png`：raw/fused/normalized 三套拍点计算出的局部 BPM 对比；
- `__probabilities.png`：raw/fused 逐帧 beat/downbeat 概率对比；
- `__grid.png`：基础 BPM、倍率决策 BPM、最终实际 BPM 和选中网格倍率；
- `__report.json`：主导 BPM 和疑似变速区间；
- 根目录 `summary.csv`：所有歌曲的 raw/fused/normalized 汇总，并包含有效时长和
  `NO_BEAT` 时长。

三套 beat CSV 都保存 `sample_index`、高精度时间、`activity_segment_id` 和
`is_no_beat`。绘图与统计不读取 Python 推理内存，而是重新加载这些 CSV、
`frames.csv`、`grid.csv` 和 `segments.csv` 后生成。

判断好坏时，优先试听 normalized，并与 raw/fused 两套 `__clicks.wav` 做 A/B：如果点击声始终落在音乐拍点上，说明结果可信；只比较全局 BPM 会漏掉相位错误、漏拍以及变速片段。

网格解码器利用离线动态规划选择倍率。高倍率网格会在理论位置附近搜索 fused logit 峰补拍，低倍率网格会比较不同相位后删拍；最后解决小于 0.25 秒的冲突并填补大于 0.5 秒的缺口，使最终相邻拍点满足 `[120, 240)`。

## 关于“瞬时 BPM”

原始瞬时值严格按照下式计算：

```text
raw_local_bpm = 60 / (当前节拍时间 - 上一个节拍时间)
```

单个漏拍会让该值减半，额外误检会让它加倍。因此脚本还输出 `smoothed_local_bpm`：它只修正孤立的疑似半速/倍速异常，再应用 5 个节拍左右的中值滤波。原始值不会被覆盖。

红色阴影表示相对主导速度偏离至少 8% 或 8 BPM、并持续至少 4 个节拍和 2 秒的候选片段。阈值可调整：

```powershell
python compare_bpm.py song.mp3 --change-ratio 0.05 --change-bpm 5 --min-change-beats 3
```

精确的 2 倍/半速变化存在音乐层级歧义，例如 70 与 140 BPM 可能描述同一律动。自动标注只能作为候选，最终应结合点击声和 `raw_local_bpm` 判断。
