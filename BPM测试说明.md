# Beat This! BPM/节拍分析

脚本 `compare_bpm.py` 使用 `Beat This!` 预训练 Transformer 输出 beat 和 downbeat，
并生成局部 BPM、疑似变速区间和试听用 click-track。脚本同时保留官方分块基线，
并使用重叠窗口融合逐帧 logits，以减少约 30 秒分块处的硬接缝。

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

## 输出怎么看

每首歌有一个独立子目录，其中包含：

- `__beat-this-raw__beats.csv`：官方 30 秒分块、`keep_first` 拼接后的拍点；
- `__beat-this-fused__beats.csv`：重叠窗口 Hann 加权融合后的拍点；
- 对应的 `__clicks.wav`：raw/fused 两套试听文件，高音点击代表 downbeat；
- `__beat-this__frames.csv`：50 FPS 的 raw/fused beat、downbeat logits 和概率；
- `__tempo.png`：raw/fused 两套拍点计算出的局部 BPM 对比；
- `__probabilities.png`：raw/fused 逐帧 beat/downbeat 概率对比；
- `__report.json`：主导 BPM 和疑似变速区间；
- 根目录 `summary.csv`：所有歌曲的 raw/fused 汇总。

判断好坏时，优先 A/B 试听 raw/fused 两套 `__clicks.wav`：如果点击声始终落在音乐拍点上，说明结果可信；只比较全局 BPM 会漏掉相位错误、漏拍以及变速片段。

重叠融合只处理分块接缝，并不会自动修复半速/倍速层级。层级修复属于下一阶段的软时序解码。

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
