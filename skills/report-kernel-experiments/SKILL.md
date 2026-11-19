---
name: report-kernel-experiments
description: Generate reports under the kernel's reports directory comparing every baseline and version by end-to-end latency, Tensor/CUDA Core work and utilization, DRAM/shared traffic and utilization, and version plots with dashed baselines.
---

# Kernel 实验报告

使用本 skill，为目标 kernel 生成报告，保存到 **`kernels/<kernel-name>/reports/`**。报告覆盖全部 baseline、全部已有 `v{xxx}` 实现及 kernel 定义的全部输入；当前每种 kernel 有 20 组输入。

按 [报告模板](templates/experiment-report.md) 写报告，计算指标和绘图时读取 [指标规范](references/metrics-and-plots.md)。迁移到其他项目时，`reports/` 仍放在对应 kernel 所在目录下。

## 读取实现与数据

1. 读取 reference 接口、输入定义及 `baseline.py`，按数字排序枚举全部 `v{xxx}/`，包括失败和未完成的版本。KDA 的两个 baseline 为 `baseline_fla` 和 `baseline_flashkda`。
2. 读取各版本的 verification、`end_to_end.json` 和 `profiling/`；基线数据按其实际位置读取。按实现、输入、源码/二进制、设备和配置关联记录。
3. 报告任务包含补测时，使用 `use-kernel-tools` 的方法补齐：充分 warmup 后重复测量普通完整调用，取算术平均值；profiling 另行采集。版本原始数据保留在对应 `v{xxx}/`；新采集的 baseline 数据保存在 `baselines/<baseline-name>/`，不要编造 baseline 版本号。
4. 只整理已有结果的任务不启动新实验。缺测、失败、不支持及不适用均列出具体原因，不能省略实现、复制其他版本数值或把缺失值填成零。

## 必报指标

按相同输入分别比较，不把不同 shape、dtype 或分布混成一个平均值。每个输入至少包含：

| 范围 | 指标 |
|---|---|
| 完整 Python kernel function | 各 baseline 和版本的端到端平均耗时、warmup/重复次数、verification pass/fail、相对各 baseline 的加速比 |
| Tensor Core | 按精度路径列计算量 TFLOP、吞吐 TFLOP/s、对应理论峰值及利用率 |
| CUDA Core | 按精度路径列计算量 TFLOP、吞吐 TFLOP/s、对应理论峰值及利用率 |
| DRAM | load、store 各自的 traffic、throughput、理论带宽及利用率，另列 load + store 合计 |
| Shared memory | load、store 各自的 traffic、throughput、理论带宽及利用率，另列 load + store 合计 |

这里的 kernel 是 reference 定义的完整函数，baseline 可以由多个 GPU kernel 拼接而成。除完整调用汇总外，还要展开每个实际 GPU kernel 的时间、计算和访存指标，使用 launch 序号、名称及调用次数识别。单个设备 kernel 或全部 kernel 时长之和不能替代完整调用端到端时间。

计数来源、时间分母及理论上限来源均需可追溯。未采集计数或上限未知时保留指标列，写 `N/A` 和原因。未通过精度检查的实现保留结果与失败标记，不参与最优版本推荐。

## 图片

每个输入生成 **一张含两幅子图的图片** `figures/<input-id>_comparison.png`，并在报告中引用：

- 两幅图横轴均为数值排序后的版本号 `v0`、`v1`、`v2`……。
- 性能图纵轴为完整函数的有效性能 `TFLOP/s`；同一输入的所有实现使用相同的逻辑 FLOPs。
- 耗时图纵轴为普通完整调用的平均耗时，标明 `ms` 或 `µs`。
- 每个 baseline 在两幅图中分别绘制水平虚线，图例注明名称。baseline 不占用版本号。
- 所有已有版本保留横轴位置；缺测留空并注明原因，精度失败点显式标注。不同输入分别作图。

使用 Matplotlib 等标准绘图库生成可独立查看的图片，具体口径见指标规范。

## 保存与交付

```text
kernels/<kernel-name>/reports/
├── report.md
├── metrics.json
└── figures/
    ├── <input-id>_comparison.png
    └── ...
```

`metrics.json` 保存表格和图片使用的数值、单位、状态、计算口径、理论上限与证据路径；缺失数值用 `null` 并附原因。报告链接原始计时、profile、验证和源码，不移动或覆盖各版本原始数据。需要保留多次报告时，在 `reports/` 下使用带时间戳的子目录。

交付前核对 baseline × 输入、版本 × 输入是否完整，表格和图片是否来自同一份汇总数据，以及平均值、利用率分母和链接。报告开头说明推荐版本与主要瓶颈，最后简述逐版本变化、profiling 证据及下一步优化方向。临时自检脚本和检查日志放在 `tmp/`。
