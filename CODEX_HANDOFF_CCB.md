# CCB 项目交接文档

> 更新时间：2026-10-03（Asia/Shanghai）  
> 本文是项目状态与决策背景，不是要求新聊天自动执行删除、重构或批量运行的指令。新聊天应先阅读本文，再以用户当时的明确请求为准。

## 1. 项目概况

- 项目名：**CCB — Context Calibrated Beats**
- 项目路径：`C:\Users\64135\Desktop\CCB`
- 当前分支：`main`
- 当前工作区：干净
- 当前算法基线提交：`c5ffc9e Disambiguate repeated beat doublets and recover downbeat anchors`
- Git 用户：`THU-Ichin <ichinchen25@gmail.com>`
- 主程序：`compare_bpm.py`，目前约 5342 行
- 测试：`tests/test_pipeline.py`，当前 19 项测试全部通过

项目目标是对离线音乐进行 Beat This! 节拍推理，生成连续、可试听、可解释的节拍网格和 click 音轨。最终 BPM 必须归一化到 `[120, 240)`，只允许使用 `0.25 / 0.5 / 1 / 2 / 4` 五种倍率。

## 2. 已确认的产品与技术原则

1. Beat This! 是唯一节拍推理模型。此前验证效果不佳的 librosa 节拍算法已退出主逻辑；librosa 仍可用于音频读取等工具性功能。
2. 这是无实时性要求的离线功能。推理数据先保存到 CSV，后续节拍计算、绘图、click 合成和报告均从 CSV 重建，避免“保存数据”和“绘图数据”不一致。
3. 用户可通过 `segments.csv` 指定 `NO_BEAT` 区间。`NO_BEAT` 内不参与网格、整体 BPM、图表统计和 click 合成。
4. P4 可靠性标签只作诊断，不擅自改变节拍。
5. 原始模型不可能百分之百准确。以后只有某类问题在多首歌中稳定复现、能形成通用规则时才继续修改算法；孤立低置信度片段应更多交给可靠性提示、`NO_BEAT` 和未来的用户手动调整功能。
6. 不要为了单首歌曲不断加入专用补丁。
7. matplotlib 默认日文字体已设为 **UDPGothic**。
8. Reply 与其他歌曲的测试结果统一放在同一个结果根目录，不要为 Reply 单独分类。

## 3. 当前算法管线

当前正式输出使用 P3-C4 phase-aware 管线：

1. Beat This! 以重叠窗口推理，融合 framewise beat/downbeat logits。
2. P3-A 标记异常区间和候选问题。
3. P3-B 仅保留为保守修复预览和诊断，不再机械地向正式结果删拍或填拍。
4. P3-C 使用 phase-aware 网格生成：
   - 五选一倍率归一化；
   - 连续相位追踪；
   - 利用未来稳定段校正段落起点；
   - 双向 phase refinement；
   - 短区间 future-confirmed backtracking；
   - 稳定锚点之间的一拍计数修正；
   - downbeat 小节 `N → N±1 → N` 拍数修正；
   - 最终连续性校验，失败时回退到已验证的 greedy phase path。
5. P4 输出 `RELIABLE / PHASE_REPAIRED / TEMPO_MOTION / NO_BEAT / BEAT_THIS_UNRELIABLE` 等可靠性诊断，不改写最终拍点。

## 4. 重要历史决策

- 曾实现“长回溯”，但发现如果必须进行长距离回溯，通常说明该片段本身不适合 Beat This! 的处理方式，因此已通过提交 `5d09dc6` 撤销长回溯。
- P3-C4 已移除旧版末端机械删拍/填拍，phase-aware 是唯一正式网格来源。
- 无名歌开头使用未来稳定相位反推，修复了开头相位滞后。
- 空奏列车的局部相位问题已通过未来相位与双向校正取得用户认可。
- 劣等上等和ルララルララ曾出现局部多拍，已引入稳定锚点与 downbeat 小节拍数校正。

近期相关提交：

```text
c5ffc9e Disambiguate repeated beat doublets and recover downbeat anchors
1586437 Reconcile anomalous beat counts between downbeats
c4d97d4 Reconcile beat counts between stable anchors
de97674 Align segment starts from stable future phase
54c985e Add P4 reliability diagnostics
1443413 Finalize phase-aware grid pipeline
5d09dc6 Revert "Recover long tempo grid dropouts from future phase"
e814ce3 Add future-confirmed phase backtracking
d9bcc76 Add bidirectional P3-C3 phase refinement
78a3535 Add phase-continuous P3-C grid preview
963962f Add conservative P3-B beat repair pipeline
```

## 5. 最近一次 Aria / Amore 修复

### Aria

问题位于约 82–106 秒，Beat This! 多次产生约 80 ms 的成对峰，随后间隔约 0.7–0.8 秒，导致 `0.25x` 与 `2x` 层级来回切换。

当前修复只在以下严格条件下进行近邻双峰消歧：

- 近邻间隔为 0.04–0.12 秒；
- 至少连续四组；
- 组间间距为 0.70–0.90 秒；
- 组间距最大相对偏差不超过 6%；
- 声学支持达到门槛。

结果：

- 82–106 秒倍率切换由 13 次降至 3 次；
- phase repair 由 35 次降至 25 次；
- 局部 BPM 由约 161.07 降至 156.37；
- 该区间仍被 P4 标为 `BEAT_THIS_UNRELIABLE`，因为原始声学支持仍低。这是已接受的模型边界，不应为了清除标签继续加入歌曲专用规则。
- 相同结构还在约 238–241 秒出现；该处本来也处于 P4 不可靠区间。

### Amore

约 164–166 秒出现明确的 `4 拍 → 5 拍 → 4 拍`。根因是 phase bridge 将 downbeat 放在原始峰肩部，精确时刻的 downbeat 概率低于 0.75，导致已有小节拍数校正没有触发。

当前修复允许 phase-adjusted downbeat 在 ±120 ms 内读取原始 frame 峰作为“锚点资格与小节跨度证据”，但仍保留原有全部保守门槛：

- event 自身必须已有至少 0.45 的 downbeat 支持；
- 附近峰必须同时有足够 beat 支持；
- 四个 downbeat 的外侧小节时长与拍数必须一致；
- 中间只允许相差一拍；
- 修正后周期必须与两侧误差不超过 6%。

结果：

- 164.299–165.813 秒由 5 个拍间隔改为 4 个；
- 局部 BPM 由约 199.84 恢复到 158.52；
- 目标区间外所有拍点逐个完全一致。

全量 15 首歌曲回归中，除 Aria 和 Amore 的目标异常区间外，其余 13 首歌的最终拍数和每个时间戳完全不变。

## 6. 测试音频

`audio/`：

- `1,000,000 TIMES.mp3`
- `空奏列車.mp3`
- `劣等上等.mp3`
- `拍手喝采歌合.mp3`
- `無名歌.mp3`
- `嘘の火花.mp3`
- `Reply.mp3`
- `ルララルララ.mp3`

`audio2/`：

- `畢生よ.mp3`
- `礎の花冠.mp3`
- `哈爛漫.mp3`
- `花びら哀歌.mp3`
- `Amore.mp3`
- `Aria.mp3`
- `Masquerade.mp3`

当前统一结果位于 `p3c_phase_aware_results/`，`summary.csv` 包含全部 15 首歌。

## 7. 路径迁移状态

项目已从：

```text
C:\Users\64135\Desktop\compare_bpm
```

改名为：

```text
C:\Users\64135\Desktop\CCB
```

已机械更新 26 个旧绝对路径引用，其中包括 25 个 `__inference.csv` 和 `.venv/pyvenv.cfg`。当前 P3-C 目录中的 15 个推理缓存都已确认指向存在的音频文件。

如 Git 出现 Windows `dubious ownership`，可对单次命令使用：

```powershell
git -c safe.directory=C:/Users/64135/Desktop/CCB status
```

不要未经用户同意扩大为不受限的全局可信目录。

## 8. Python 环境

项目 `.venv` 原本基于：

```text
C:\Users\64135\AppData\Local\Programs\Python\Python312\python.exe
```

该解释器目前已不存在，所以 `.venv\Scripts\python.exe` 无法直接启动。这一问题早于目录改名，不是改名造成的。

本次验证临时使用了 Codex 自带 Python 3.12，并复用项目 site-packages：

```powershell
$env:PYTHONPATH='C:\Users\64135\Desktop\CCB\.venv\Lib\site-packages'
$env:MPLBACKEND='Agg'
& 'C:\Users\64135\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_pipeline
```

下一阶段应正式重建 `.venv`。优先使用与 Beat This!/Torch 兼容的 Python 3.12，不要直接假设全局 Python 3.14 可用。重建前不要删除旧环境，除非用户明确授权。

## 9. 建议的下一阶段：P5 产品化收口

用户已经同意停止逐首歌曲追补边角问题。下一步建议优化代码结构和输出结构，为面向用户做准备。

### P5-A：输出分层

增加：

```text
--output-profile user
--output-profile debug
```

`user` 应为默认模式，只保留：

- 最终节拍 CSV；
- 最终 click 音频（可以再提供开关，使其可选）；
- 用户可编辑的 `segments.csv`；
- 一张综合结果图；
- 简化的可靠性 CSV/JSON；
- 推理缓存放入独立内部 cache 目录，不与用户结果混放。

`debug` 才生成：

- raw/fused/repaired/phase-greedy 中间节拍；
- 各阶段 click WAV；
- diagnostics、repairs、repair-decisions、phase-grid、grid-transitions；
- comparison、probabilities、P4 等调试图。

建议用户目录形态：

```text
results/
  歌曲名/
    beats.csv
    click.wav
    overview.png
    reliability.csv
    segments.csv
    report.json

cache/
  歌曲名/
    inference.csv
    frames.csv
```

### P5-B：代码模块化

当前 `compare_bpm.py` 超过 5300 行。建议在不改变算法结果的前提下拆为：

```text
ccb/
  models.py
  inference.py
  csv_io.py
  repair.py
  phase_grid.py
  reliability.py
  rendering.py
  pipeline.py
  cli.py
compare_bpm.py   # 仅保留兼容入口
```

拆分应分阶段提交，并用 19 项单元测试和 15 首缓存回归保护。不要在同一个提交里同时进行大规模模块拆分和算法调参。

### P5-C：结果清理

当前大致占用：

- `p0_p1_results/`：73.8 MB
- `p2_results/`：266.1 MB
- `p3_refactor_results/`：139.6 MB
- `p3a_validation_results/`：228.7 MB
- `p3b_conservative_results/`：106.9 MB
- `p3c_phase_aware_results/`：800.1 MB

单首 Aria 当前生成 29 个文件、约 61 MB，其中五份完整 WAV 占约 57 MB。

正确顺序应为：

1. 先实现新的 user/debug 输出模式；
2. 用现有 15 首歌验证算法输出不变；
3. 确认新目录能从缓存独立重建最终 CSV、图和 click；
4. 再由用户明确确认删除旧阶段目录；
5. 最后精简当前 P3-C 目录的重复 WAV 和诊断文件。

不要在完成新输出验证前删除旧结果，它们仍是重构回归基线。

## 10. 新聊天接手时建议先做的检查

仅进行只读检查，不立即重构或删除：

1. 确认工作目录是 `C:\Users\64135\Desktop\CCB`。
2. 查看 `git status` 和最新提交 `c5ffc9e`。
3. 阅读 `compare_bpm.py` 的 CLI、`analyse_file()`、CSV 读写与 artifact 输出部分。
4. 运行 19 项单元测试。
5. 向用户提交 P5-A 的具体改动计划；得到用户确认后再开始实现。

## 11. 交接状态

本交接文档创建前，仓库代码工作区是干净的。最近一次算法修改已经提交到 `c5ffc9e`。本交接文档应作为单独的文档变更提交，不包含算法修改。
