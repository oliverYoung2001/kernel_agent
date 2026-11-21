# KDA profiling

在仓库根目录激活实验环境，并用 `KDA_EXPERIMENT` 选择实验。所有采集命令通过 `kernels.kda_megakernel.run` 执行。

| 模式 | 用途 |
|---|---|
| `--mode timing` | 预热后计时，保存完整调用的普通耗时样本 |
| `--mode profile` | 捕获一次完整调用的 CUPTI 时间线 |
| `--mode capture` | 由 NCU 启动，在预热后的完整调用边界开启和停止采集 |
| `--mode suite --hardware` | 针对所选实现和 case 批量执行验证、普通计时、CUPTI 和 NCU 采集 |

在分配到 GPU 的计算节点运行 suite。它从 PyTorch 当前 CUDA 设备读取型号、UUID、计算能力、SM 数和显存，写入测量记录。指定 `--device cuda:1` 时，验证、计时、CUPTI 和 NCU 子进程都使用作业可见的第 2 张卡。

`--hardware` 会先在选中的设备上查询 NCU 支持的完整指标名。耗时是必需项，其他缺项保存在 collection 的 `unavailable_ncu_metrics` 中。缺少计数时，对应吞吐和利用率显示为 `N/A`，其余测量照常输出。半精度 SASS 归因仅在识别到 Hopper 且能解码实际指令时使用。

报告沿用采集时保存的峰值规格。内置规格覆盖 H100 PCIe 80 GB 和 H20 96 GB 整卡，并核对型号、SM 数和容量。其他型号或规格可在实验的 `experiments/` 或 `environment/` 放置 `hardware*.json`，以 `model` 和可选的 `sm_count`、`memory_GB_advertised` 匹配。缺少匹配规格时保留实测吞吐。在登录节点离线生成报告时无需可见 GPU。

experiment0 的 LACP 同卡对照包含[计时样本和逐 kernel 数据](../experiment0/reports/single-call-lacp/results.json)。从该次采集的原始 manifest 重建报告时，如果实验只有一份 collection，可以省略 `--tag`。

```bash
manifest=kernels/kda_megakernel/cases/cases_0.json  # 指向从归档恢复的原始 manifest
KDA_EXPERIMENT=experiment0 python -m kernels.kda_megakernel.profiling.make_report --manifest "$manifest"
```
