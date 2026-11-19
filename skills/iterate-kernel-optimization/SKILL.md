---
name: iterate-kernel-optimization
description: Implement a first megakernel as v0, create a new v{xxx} directory for each iteration, then profile, measure mean end-to-end time, and retain the code and corresponding data together.
---

# 最简 megakernel 优化循环

## 首轮：从 reference 实现 megakernel

读取 kernel 的 Python 接口、reference 和输入集合，据此实现第一个 megakernel 版本，代码保存到 `kernels/<kernel-name>/v0/`。保持相同的输入/输出接口和计算语义，使用 `kernel-verification` 对每种输入比较 reference 的所有输出，得到逐输入 pass/fail。

首版完成后，也按下面的第 2、3 步采集性能并分析，作为后续改进的起点。

## 后续每轮

### 1. 改进当前版本实现

根据上一轮分析改进当前 megakernel。每轮开始时新建一个 `v{xxx}` 文件夹，版本号从 0 递增：`v0`、`v1`、`v2`……。从上一版源码创建本轮实现，在新目录中修改，不覆盖已有版本，也不把上一版测量结果复制成本轮结果。

简要记录改了什么。改动后使用 `kernel-verification` 检查全部输入的输出精度；有 fail 时先修复，再把该版本用于有效性能比较。

### 2. Profiling 和端到端性能测试

使用 [use-kernel-tools](../use-kernel-tools/SKILL.md) 中描述的合适工具，对当前版本做 profiling。

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

写清楚本轮结果说明了什么、下一轮准备修改哪里及原因，然后回到第 1 步继续迭代。

## 每轮记录

使用 [简要迭代记录](references/iteration-record.md) 记录实现变化、逐输入 pass/fail、profiling 发现、端到端平均耗时和下一步改进。本轮代码、实际 profiling 数据及端到端数据保存在对应的 `v{xxx}` 文件夹中。

仅供 agent 使用的临时自检脚本、日志和检查产物放到 `tmp/`。生成代码的注释使用英文。
