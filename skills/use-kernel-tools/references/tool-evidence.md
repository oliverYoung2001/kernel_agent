# Profiling 与端到端数据记录

所有记录归属采集时使用的 `v{xxx}` 目录。输入 ID 对应 kernel 已定义的具体输入，避免不同 shape/dtype 的结果混在一起。

## Profiling

| 内容 | 记录要求 |
|---|---|
| 版本 | `v0`、`v1` 等及实际实现入口 |
| 工具 | NCU、CUPTI 或 timestamp 插桩，工具版本/插桩开关 |
| 输入 | case ID、shape、dtype 与需要的 scalar 参数 |
| 命令 | 可复现的采集命令与工作目录 |
| 范围 | kernel/launch 筛选、全部 grid 或选定 CTA/事件范围 |
| 原始结果 | 本版本 `profiling/` 内的报告、activity、timestamp 及采集日志 |
| 分析 | 关键指标、时间区间、主要耗时/等待及下一步优化线索 |

指标保留原始名称、单位和统计范围。timestamp 额外记录事件定义、时间基准和 writer；插桩结果与普通端到端计时分开。

为实验报告采集时，覆盖完整函数内的全部设备 kernel，保留 launch 身份及次数、各自时间、分精度的 Tensor/CUDA Core 操作数，以及 DRAM/shared 的 load/store 字节。记录计数是否包含 TMA、Tensor Core shared 访问、重放等，缺失指标注明原因；只有 timestamp/activity 时不能声称已经取得计算和访存计数。多 pass/replay 的计数还原为每次函数调用，不重复相加。记录设备型号、时钟、功耗及理论上限来源，供报告计算吞吐和利用率。

## 端到端性能

本版本的 `end_to_end.json` 至少包含：

- 当前版本、设备、运行命令与完整调用计时边界。
- 每个输入 ID 对应的 warmup 次数和正式重复次数。
- 每次正式调用的耗时、统一时间单位及算术平均值。
- 如测量 reference 或上一版，对应实现标识及同输入下的平均耗时。

平均值为 `sum(samples) / len(samples)`。warmup 和 profiling 运行不混入 samples；只采集成功完成且覆盖完整调用的正式样本。

本轮实际数据留在 `v{xxx}`。临时自检文件放 `tmp/`，不把版本的正式测量数据移到那里。

baseline 使用相同输入和计时边界；新采集的 baseline 原始数据放在 kernel 的 `baselines/<baseline-name>/` 下，保留 `profiling/` 和 `end_to_end.json`，不占用版本号。报告是从这些数据生成的汇总，保存在 kernel 的 `reports/` 下。
