---
name: report-kernel-experiments
description: Generate reports under the current experiment's reports directory comparing every baseline and version by end-to-end latency, Tensor/CUDA Core work and utilization, DRAM/shared traffic and utilization, and version plots with dashed baselines.
---

# Kernel 实验报告

为目标 kernel 生成实验报告，覆盖全部 baseline、全部已有 `v{xxx}` 实现及 kernel 定义的全部输入；当前每种 kernel 有 20 组输入。

按 [报告模板](templates/experiment-report.md) 写报告，计算指标和绘图时读取 [指标规范](references/metrics-and-plots.md)。

## 读取实现与数据

1. 读取 reference 接口、输入定义及 `baseline.py`，按数字排序枚举全部 `v{xxx}/`，包括失败和未完成的版本。KDA 的两个 baseline 为 `baseline_fla` 和 `baseline_flashkda`。
2. 读取各版本的 verification、`end_to_end.json` 和 `profiling/`；基线数据按其实际位置读取。按实现、输入、源码/二进制、设备和配置关联记录。
3. 报告任务包含补测时，使用 `kernel-profiling` 的方法补齐：充分 warmup 后重复测量普通完整调用，取算术平均值；profiling 另行采集。版本原始数据保留在对应 `v{xxx}/`；新采集的 baseline 数据保存在 `baselines/<baseline-name>/`，不要编造 baseline 版本号。
4. 只整理已有结果的任务不启动新实验。缺测、失败、不支持及不适用均列出具体原因，不能省略实现、复制其他版本数值或把缺失值填成零。

## 必报指标

**主报告 `reports/report.md` 必须同时呈现完整调用汇总和逐 launch 分析。对每个输入，分别为每一个 baseline 和每一个已有版本列出一次完整调用中的全部实际 GPU kernel。同名 kernel 多次发射也按原始顺序逐次列出。** 同一输入内比较各实现，分别列出以下指标。

| 范围 | 指标 |
|---|---|
| 完整 Python kernel function | 各 baseline 和版本的端到端平均耗时、warmup/重复次数、verification pass/fail、相对各 baseline 的加速比 |
| Tensor Core | 按精度路径列计算量 TFLOP、吞吐 TFLOP/s、对应理论峰值及利用率 |
| CUDA Core | 按精度路径列计算量 TFLOP、吞吐 TFLOP/s、对应理论峰值及利用率 |
| DRAM | load、store 各自的 traffic、throughput、理论带宽及利用率，另列 load + store 合计 |
| Shared memory | load、store 各自的 traffic、throughput、理论带宽及利用率，另列 load + store 合计 |

**逐 kernel 分析的每张表都以 `global` 为第一条数据行**，序号栏原样填写小写 `global`，随后按采集顺序列出各次 launch。`global` 汇总当前输入、当前实现的一次调用：累加发射次数、设备时间、分精度工作量和分层读写流量，以汇总工作量或流量除以累计设备时间计算吞吐，再按相应峰值计算利用率。算术强度使用总 FLOPs 除以总 DRAM 字节。

完整调用以 reference 定义的函数为边界。逐 launch 表使用实现名、launch 序号、实际 kernel 名称和调用次数识别每次发射，列出设备时间，以及上表中各精度的计算指标和 DRAM、Shared 的读、写、合计指标。逐 launch 吞吐使用该 launch 自身的设备时间。对每种实现指出主要耗时和访存来自哪些 launch，给出占比和分母。明细文件、原始 trace 和 JSON 供追溯，正文仍须展开全部 launch。缺少捕获时，在对应输入与实现处标明缺失范围及原因。

计数来源、时间分母及理论上限来源均需可追溯。未采集计数或上限未知时保留指标列，写 `N/A` 和原因。未通过精度检查的实现保留结果与失败标记，不参与最优版本推荐。

## 图片

每个输入生成 **一张含两幅子图的图片** `figures/<input-id>_comparison.png`，并在报告中引用：

- 两幅图横轴均为数值排序后的版本号 `v0`、`v1`、`v2`……。
- 性能图纵轴为完整函数的有效性能 `TFLOP/s`；同一输入的所有实现使用相同的逻辑 FLOPs。
- 耗时图纵轴为普通完整调用的平均耗时，标明 `ms` 或 `µs`。
- 每个 baseline 在两幅图中分别绘制水平虚线，图例注明名称。baseline 不占用版本号。
- 所有已有版本保留横轴位置；缺测留空并注明原因，精度失败点显式标注。不同输入分别作图。

使用 Matplotlib 等标准绘图库生成可独立查看的图片，具体口径见指标规范。

## 文件管理

生成的报告统一放在当前实验的 `experiment{xxx}/reports/` 文件夹中。主报告、汇总指标、图表及本实验专用的报告采集入口与生成脚本均放在该目录下。报告需要补采时，入口调用 `{kernel_name}/profiling/` 中的采集实现或转发到 `{kernel_name}/run.py`。

```text
kernels/<kernel-name>/experiment{xxx}/reports/
├── report.md
├── metrics.json
└── figures/
    ├── <input-id>_comparison.png
    └── ...
```

`metrics.json` 保存表格和图片使用的数值、单位、状态、计算口径、理论上限与证据路径；缺失数值用 `null` 并附原因。报告链接原始计时、profile、验证和源码，不移动或覆盖各版本原始数据。需要保留多次报告时，在 `reports/` 下使用带时间戳的子目录。

## 交付检查

交付前逐一核对每个输入、每种实现的原始 launch 数、顺序和名称与主报告中的设备时间、计算、访存三组记录是否一致。核对表格、图片、平均值、利用率分母和链接。报告开头说明推荐版本与主要瓶颈，最后简述逐版本变化、profiling 证据及下一步优化方向。临时自检脚本和检查日志放在本实验的 `tmp/`。
