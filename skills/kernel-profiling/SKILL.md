---
name: kernel-profiling
description: Profile kernel versions using NCU, CUPTI, or intrusive timestamp instrumentation, and retain profiling and end-to-end timing data alongside the corresponding code in its v{xxx} directory.
---

# Kernel profiling

kernel agent 可以使用 **NCU、CUPTI、侵入式 timestamp 插桩** 分析当前 megakernel。按本轮优化问题选择合适工具，不要求每轮使用全部工具。

## 工具与用途

| 工具 | 用途 | 保留的数据 |
|---|---|---|
| NCU（Nsight Compute） | 分析计算、访存、资源使用和主要瓶颈 | 原始报告、导出指标、采集命令和日志 |
| CUPTI | 采集 kernel/activity 时间线、launch 与执行时间等运行信息 | 原始 activity/trace 数据、关联信息、采集命令和日志 |
| 侵入式 timestamp 插桩 | 在 kernel 内记录指定阶段或事件的时间，分析等待、重叠和任务执行情况 | 插桩代码/开关、原始 timestamp、事件定义和解析结果 |

具体参数与支持指标按已安装工具版本核对。生成代码的注释使用英文。

## profiling代码入口

每个 kernel 的 profiling 命令统一从 `{kernel_name}/run.py` 进入。`run.py` 只声明命令行参数、选择实验并分发模式。使用与普通运行相同的实现名和 case ID，保证采集覆盖同一份输入上的完整调用。

以 `kda_megakernel` 为例，从仓库根目录运行以下命令可采集 experiment1 中 v0 的 CUPTI 时间线。

```bash
export KDA_EXPERIMENT=experiment1
python -m kernels.kda_megakernel.run --mode profile --impl v0 \
  --case kda_kimi_linear_48b_a3b_12 --tag=-review
```

NCU 使用 `--mode capture` 作为捕获目标。以 `ncu --profile-from-start off` 启动该命令，目标在 warmup 完成后开启 profiler 并执行一次完整调用。批量采集使用 `--mode suite --hardware`，显式指定 `--impl`、`--cases` 和本次采集的唯一 `--tag`。

## profiling代码生成

每个 kernel 在所有 `experiment{xxx}/` 的外部共用一个 `{kernel_name}/profiling/` 源码目录。在这个目录维护 CUPTI 时间线采集、NCU 捕获边界与命令、计数器选择、SASS 与 trace 解析、timestamp 插桩驱动，以及配套的计时脚本和使用说明。`tools/` 只存放通用工具。新增采集方式时，在 `profiling/` 实现，并在 `run.py` 注册对应的入口。

实验迭代和报告脚本可调用这里的采集函数或转发到 `run.py`。采集产物按实验和实现归档，版本数据写入 `experiment{xxx}/v{xxx}/profiling/`，baseline 数据写入 `experiment{xxx}/baselines/<baseline-name>/profiling/`。跨实现的报告证据随报告保存在 `experiment{xxx}/reports/`。

## 采集与端到端测试

1. 确认当前版本目录、实现入口和具体输入 case，再运行工具。
2. 保存使用的工具、版本、命令、输入 ID、采集范围与原始结果，确保数据对应本轮代码。
3. 独立运行普通端到端测试：充分 warmup 后重复足够次数，记录每次耗时和算术平均值。时间范围为完整 Python function 调用到全部输出可用，GPU 异步工作需要正确同步。
4. profiling 与普通端到端测试分开执行。正式端到端结果使用未开启 profiler 和 timestamp 记录的普通实现。

侵入式插桩可能改变性能。保留普通实现、插桩修改和记录开关；需要时比较记录关闭/开启的结果。说明每个 timestamp 对应的事件、时间基准、writer 和单位；记录区间可能包含等待，不能直接当成硬件 busy 时间。

## 保存到对应版本目录

初始实现为 `v0`，后续为 `v1`、`v2`……。本轮 profiling 和端到端数据必须与本轮代码放在**同一个 `v{xxx}` 目录**，例如：

```text
kernels/<kernel-name>/experiment{N}/
├── v0/
│   ├── implementation.py
│   ├── kernels.cu
│   ├── profiling/
│   │   ├── ncu/
│   │   ├── cupti/
│   │   └── timestamps/
│   └── end_to_end.json
└── v1/
    ├── implementation.py
    ├── kernels.cu
    ├── profiling/
    └── end_to_end.json
```

源码名称按实际实现确定，只生成本轮使用的工具对应的目录。多次采集使用不同文件名或子目录保留，不能覆盖其他版本的数据，也不能把上一版结果复制成当前版结果。

`end_to_end.json` 按输入保存 warmup 次数、正式重复次数、各次耗时、时间单位和平均值，以及设备、计时边界和运行命令。profiling 原始数据及相关采集日志保留在本版本的 `profiling/` 下。字段说明见 [数据记录](references/tool-evidence.md)。

仅供 agent 自检的临时脚本、日志和检查文件放在本实验的 `tmp/`；本轮实际 profiling 和端到端数据保存在对应版本目录中。
