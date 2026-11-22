---
name: iterate-kernel-optimization
description: Optimize a GPU megakernel for at least 10 iterations starting at v0, verify and profile each version, measure mean end-to-end timing, analyze results, and finish with one report-kernel-experiments report.
---

# 最终优化目标

**尽力将实现的执行时间优化到接近0s!**

# 最简 megakernel 优化循环

## 第一步：读取并遵循目标 kernel 的“性能优化指导”

开始或恢复实验时，在选择后端、编写代码或提交作业前，逐个打开当前工作树的 `kernels/README.md`。按标题文字定位“性能优化指导”（`#`、`##` 等层级均可），读取正文及子节，直到下一个同级或更高层级的标题或文件末尾。同时读取“优化注意点”中的数值假设和验证要求。

在首版迭代记录中标明 README 路径，列出该章节要求的实现语言、融合范围、允许使用的库等实际约束，并说明方案如何满足。区分“必须”和“可以”。如果 README 或章节缺失、为空或无法读取，明确指出具体路径和标题。用户已在当前对话提供指导原文时，以该原文为依据并注明来源；否则先请用户提供正文或正确路径，再开始实现。

这些约束适用于 `v0` 及每个后续候选。每轮动手前核对本轮改动是否符合指导，README 更新后重新读取。例如，若要求完整 forward 实现为单个 CUDA megakernel，方案就必须把全部计算放进一次 CUDA kernel launch；若允许内部使用 CUTLASS，可使用其设备端组件。已有 reference、baseline 和历史版本的后端选择不构成例外。

## 每次独立实验的目录

每次从头运行一轮 kernel agent 实现，在目标 kernel 目录新建 `experiment{N}/`。首次使用 `experiment0/`，以后取已有实验的最大数字编号加一。创建成功后将本次编号传给后续 runner、profiling 和报告任务。恢复同一实验、补测或重建报告时使用原编号。

每个实验在自己的目录中从 `v0/` 开始计数。版本源码、baseline 测量、精度结果、计时、profiling、环境记录、报告和临时诊断均写在该实验下。共享的 reference、baseline 接口、cases 和运行工具保存在 kernel 根目录。

## 最少迭代轮数

每个优化任务至少完成 **10 轮 iterations**。首版 `v0` 计为第 1 轮，从零开始时至少完成 `v0`–`v9`；用户要求更多轮时按更高轮数执行。继续同一个未完成的任务时，核对已有版本的源码和测量证据，只累计已完成轮次，并从下一个版本号继续补足。

一轮完成需要保留实际实现改动、全部输入的验证结果、实际 profiling、普通完整调用计时及结果分析；`v0` 的实现改动为从 reference 创建首版。重复计时、补测 baseline、重新生成报告或复制相同实现不计为新增轮次。未完成验证或测量的尝试保留记录，并明确尚未完成。

性能回退但完成上述流程的轮次同样计数，保留结果并分析原因；不能因为连续几轮没有提速而在第 10 轮之前宣告任务完成。精度失败的结果保留失败标记，不参与有效性能比较。资源或执行条件导致无法达到最低轮数时，报告实际完成数、剩余轮数和原因，不将任务标为已完成。

## 首轮：从 reference 实现 megakernel

读取 kernel 的 Python 接口、reference 和输入集合，据此实现第一个 megakernel 版本，代码保存到 `kernels/<kernel-name>/experiment{N}/v0/`。保持相同的输入/输出接口和计算语义，使用 `kernel-verification`，通过 `run.py --mode verify` 对每种输入与第一个 baseline 比较所有输出，得到逐输入 pass/fail。

首版完成后，也按下面的第 2、3 步采集性能并分析，作为后续改进的起点。

## 后续每轮

### 1. 改进当前版本实现

根据上一轮分析改进当前 megakernel。每轮开始时新建一个 `v{xxx}` 文件夹，版本号从 0 递增：`v0`、`v1`、`v2`……。从上一版源码创建本轮实现，在新目录中修改，不覆盖已有版本，也不把上一版测量结果复制成本轮结果。

简要记录改了什么。改动后使用 `kernel-verification` 检查全部输入的输出精度；有 fail 时先修复，再把该版本用于有效性能比较。

### 2. Profiling 和端到端性能测试

使用 [kernel-profiling](../kernel-profiling/SKILL.md) 中描述的合适工具，对当前版本做 profiling。

若目标指导要求单次 launch，对每条候选执行路径检查预热后完整 forward 的 GPU trace，在迭代记录中写明实际 kernel 名称、发射次数和 trace 位置。若发现额外的计算 launch，保留诊断并修正实现，通过后再计入完成轮数。

另外运行普通端到端性能测试。对每种输入分别测量完整 Python function 从调用到所有输出可用的时间，包含该实现每次调用必需的准备和计算；输入生成与精度比较在计时之外。

- 先完成编译/JIT，再执行足够多次 warmup，直到耗时稳定。warmup 不计入平均值。
- 正式重复足够多次，取所有有效调用耗时的**算术平均值**，不以最快一次代替平均值。
- warmup 次数和正式重复次数可配置；可从 `warmup=100`、`repeat=1000` 起步，根据单次耗时和稳定情况调整，并记录实际次数。
- GPU 异步执行时，在计时开始前和调用完成后正确同步，确保计入完整计算，而不只是 kernel 提交时间。
- 每次调用使用相同输入值与初始状态。reference、上一版和当前版在相同输入、设备和计时边界下比较。
- profiling 与普通性能测试分开执行，端到端平均耗时使用未开启 profiler 的测试结果。

按输入记录 warmup 次数、重复次数、每次耗时、平均端到端耗时，以及相对 reference/上一版的变化。profiling 数据保存到本轮 `v{xxx}/profiling/`，端到端数据保存到同一目录下的 `end_to_end.json`。

### 3. 分析结果，决定下一轮改进

分析 profiling 结果，找出当前主要耗时、等待、访存或计算瓶颈；结合端到端平均耗时判断本轮改动是否有效。

写清楚本轮结果说明了什么、下一轮准备修改哪里及原因。未达到至少 10 轮或用户指定的更高轮数时，回到第 1 步继续迭代。完成所需轮次并确定最终保留版本后，进入报告步骤。

## 每轮记录

使用 [简要迭代记录](references/iteration-record.md) 记录实现变化、逐输入 pass/fail、profiling 发现、端到端平均耗时和下一步改进。本轮代码、实际 profiling 数据及端到端数据保存在对应的 `v{xxx}` 文件夹中。

仅供 agent 使用的临时自检脚本、日志和检查产物放到本实验的 `experiment{N}/tmp/`。生成代码的注释使用英文。

## 优化完成后生成报告

针对本次优化的 kernel 和同一个 `experiment{N}`，执行一次 [report-kernel-experiments](../report-kernel-experiments/SKILL.md)。按该 skill 的要求汇总全部 baseline、版本及输入，在本实验的 `reports/` 下生成并核对 `report.md`、`metrics.json` 和逐输入对比图。核对完成后，向用户汇报全部轮次、最终保留版本，并附上报告链接。
