# <kernel-name> 实验报告

> 保存为对应 kernel 的 `reports/report.md`。尖括号为填写提示。全部 baseline、全部版本、全部输入都需要列出；缺测写 `N/A` 及原因。

## 结论与测量配置

记录时间、GPU/变体、时钟/功耗、环境、计时边界和输入集合。概述通过验证的推荐版本、相对各 baseline 的收益及主要瓶颈；没有合格结果时说明原因。

| 实现 | 源码/二进制身份 | 计时/验证/profile 证据 | 状态及缺失原因 |
|---|---|---|---|
| <每个 baseline 和每个 vN 各一行> | | | |

| 理论上限 | 值及单位 | 精度/读写/SM/频率口径 | 资料或公式来源 |
|---|---|---|---|
| <Tensor Core 各精度> | | | |
| <CUDA Core 各精度> | | | |
| DRAM | | | |
| Shared memory | | | |

## 输入 <input-id>：<shape / dtype / 分布 / seed>

以下完整小节对每个输入重复；当前每个 kernel 有 20 组输入。

### 完整函数端到端时间

| 实现 | verification | warmup / repeat | 平均耗时 (ms) | 相对 <baseline-A> 加速比 | 相对 <baseline-B> 加速比 | 状态/证据 |
|---|---|---|---|---|---|---|
| <全部 baseline、v0、v1……逐行列出> | | | | | | |

给出 reference 逻辑 FLOPs 的逐阶段公式、代入值、总量及逻辑 TFLOP/s。每个 baseline 有独立加速比列，数值为 `baseline_time / implementation_time`。

### 完整函数计算指标

分项使用实际执行工作量，吞吐时间分母为普通完整调用平均时间。

| 实现 | Tensor / CUDA Core | 执行/累加精度 | 工作量 (TFLOP) | 吞吐 (TFLOP/s) | 理论峰值 (TFLOP/s) | 利用率 (%) | 计数/峰值来源及限制 |
|---|---|---|---|---|---|---|---|
| <每个实现 × 执行单元/精度路径> | | | | | | | |

### 完整函数访存指标

| 实现 | DRAM / Shared | load / store / total | Traffic (GB) | Throughput (GB/s) | 理论带宽 (GB/s) | 利用率 (%) | 计数与分母口径/证据 |
|---|---|---|---|---|---|---|---|
| <每个实现 × 两个层级 × 三个方向> | | | | | | | |

### 各设备 kernel 明细

覆盖 baseline 全部组成 kernel 和各版实际 kernel。设备时间来自 profiling，不能替代普通完整调用时间。

| 实现 | Launch 序号 / kernel 名称 | 次数 | 平均设备时间 (µs) | 工具/证据 |
|---|---|---|---|---|
| <逐设备 kernel> | | | | |

每个设备 kernel 重复下面两表，或给出等价完整长表；吞吐分母使用其自身设备时间。

| 实现 / launch | Tensor / CUDA Core | 精度 | 工作量 (TFLOP) | 吞吐 (TFLOP/s) | 理论峰值 (TFLOP/s) | 利用率 (%) | 来源/限制 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

| 实现 / launch | DRAM / Shared | load / store / total | Traffic (GB) | Throughput (GB/s) | 理论带宽 (GB/s) | 利用率 (%) | 来源/限制 |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### 版本对比图

![<input-id>：左为有效性能，右为端到端平均耗时；baseline 为水平虚线](figures/<input-id>_comparison.png)

两幅图横轴均为版本号。每个 baseline 各绘一条虚线；说明缺测、不支持和精度失败标记。

## 逐版本分析与后续优化

| 版本 / 父版本 | 实现改进 | 计算/访存及 profile 发现 | 端到端效果 | 下一步 |
|---|---|---|---|---|
| <所有已有版本，含失败及未完成项> | | | | |

链接源码与原始数据，将证据支持的解释和待验证假设分开；不能把局部 kernel 改善直接当成完整调用收益。

## 复现与数据

链接 [机器可读汇总](metrics.json)，给出实际存在的环境入口、验证、普通计时、profile 和绘图命令，注明源码/配置、原始结果路径和生成图片的方法。原始数据保留在对应版本或 baseline 目录。
