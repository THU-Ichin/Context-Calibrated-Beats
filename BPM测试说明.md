# Beat This! BPM/节拍分析

脚本 `compare_bpm.py` 使用 `Beat This!` 预训练 Transformer 输出 beat 和 downbeat，
并生成局部 BPM、疑似变速区间和试听用 click-track。

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
python compare_bpm.py "D:\Music\song.mp3" -o bpm_results
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

## 输出怎么看

每首歌有一个独立子目录，其中包含：

- `__beats.csv`：每个节拍的位置、相邻节拍得到的原始 BPM、稳健平滑后的 BPM；
- `__clicks.wav`：原音频叠加节拍点击声；`Beat This!` 的高音点击代表 downbeat；
- `__tempo.png`：Beat This! 拍点计算出的局部 BPM 曲线；
- `__report.json`：主导 BPM 和疑似变速区间；
- 根目录 `summary.csv`：所有歌曲的总表。

判断好坏时，优先试听 `__clicks.wav`：如果点击声始终落在音乐拍点上，说明结果可信；只比较全局 BPM 会漏掉相位错误、漏拍以及变速片段。

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
