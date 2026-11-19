# 指标与绘图规范

## 范围与时间

每条记录标明输入 ID、实现、shape/dtype/分布、源码身份和设备。baseline 与版本执行相同接口和输入，全部输出都计入完整调用。

普通端到端时间为完成编译/初始化、充分 warmup 后正式样本的算术平均值：`t_call = sum(samples) / len(samples)`。保存样本、warmup/repeat、同步方法和计时边界，覆盖函数内必要的 padding、布局转换、reset 和输出处理。主机计时与 CUDA event 计时分别说明实际范围，不能混用后直接比较。

设备 kernel 时间来自独立 CUPTI/NCU 采集，记录 launch 序号、名称、次数和每次调用平均时间。同名 kernel 只有工作、精度和计数范围相同时才可合并。profile 时间不能冒充普通端到端时间；并发 kernel 时长之和也不是调用墙钟时间。

完整函数计算量和流量为一次调用内全部设备 kernel 相应计数之和。profile 与普通计时对应相同实现、输入和配置。多次 replay、采集 pass 或正式调用的计数先还原为每次函数调用，不能重复相加。

## 计算量、吞吐与利用率

工作量单位为 TFLOP，速率单位为 TFLOP/s：

```text
work_TFLOP = FLOPs / 1e12
throughput_TFLOP_s = FLOPs / time_s / 1e12
utilization_pct = throughput_TFLOP_s / theoretical_peak_TFLOP_s * 100
```

完整函数有效性能使用固定逻辑工作量：`logical_FLOPs(input) / t_call / 1e12`。写出 reference 各阶段公式和具体输入代入值，同一输入的所有实现使用同一逻辑分子。实际执行 FLOPs 另列，覆盖 padding、冗余工作和 helper，用于 Tensor/CUDA Core 分项吞吐与利用率。

- Tensor Core 与 CUDA Core 分开，分别按真实 FP16/BF16/TF32/FP32/FP64 路径统计，注明累加精度。混合精度分别比较对应峰值。
- 保留计数器原名、单位、聚合范围。已经计为浮点操作总数的计数器不再乘 2、SM 数或协作线程数。
- 指令/源码推导写出公式：普通 FMA 按乘与加共 2 FLOPs，矩阵乘加按 `2*M*N*K`，结合动态次数和有效 lane。整数、地址计算及 exp/div 等特殊函数另列，不随意套用 FMA 峰值。
- 源码逻辑计数不能伪装成硬件实测执行计数。确实未使用某单元时可记零，未采集时写 `N/A`。

设备 kernel 分项吞吐使用它自身的设备时间。完整函数汇总使用 `t_call`，标明这是包含其他阶段和调用开销的端到端有效利用率。可补充工具直接报告的 active-cycle/peak utilization，但保留完整指标名及原分母，不能与上述利用率互换。

理论峰值匹配实际 GPU 型号/变体、精度、dense/sparse 模式、SM 范围和频率。生成报告时核实厂商资料，保存来源链接及字段，或明确硬件公式与参数来源；记录实测时钟及功耗限制，不沿用其他 GPU 峰值。无匹配上限时利用率为 `N/A`。超过 100% 时核查计数、时间与分母，不截断结果。

## DRAM 与 shared memory

每个层级分别列 `load`、`store`、`load+store`，每项具有 traffic、throughput、理论上限及利用率：

```text
traffic_GB = bytes / 1e9
throughput_GB_s = bytes / time_s / 1e9
utilization_pct = throughput_GB_s / matching_peak_GB_s * 100
```

统一使用十进制 GB、GB/s；若用 GiB 则明确改为 `2**30`。完整函数与设备 kernel 的时间分母分别与计算指标一致。

DRAM 使用实际 DRAM 读写字节，或明确标注估算；global load/store 请求字节不等于经 cache 后的 DRAM traffic。shared 说明是请求字节还是实际服务流量，是否包含 bank conflict、重放、TMA 和 Tensor Core 访问；工具缺少的部分说明覆盖限制，不将 LDS/STS 字节冒充全部 shared 流量。TMA global-to-shared 的端点分别归入对应层级，不在 DRAM 重复相加。

DRAM 带宽通常为整卡总读写上限；shared 上限说明每 SM 服务宽度、频率、参与 SM 数以及读写是否共享通道。只有总峰值时，单方向利用率可按共同峰值计算，但标注“占总带宽上限”，不能捏造独立 load/store 峰值再相加。没有可靠 shared 理论值时保留实测 traffic/throughput，峰值和利用率写 `N/A` 及原因。

## 图与汇总数据

每个输入一张左右两幅子图的图片：左侧有效性能 TFLOP/s，右侧完整调用平均耗时。版本按数字排序，`v2` 在 `v10` 之前；全部已有版本保留位置，缺测不能画成零或连线跨过缺失点。精度失败但有计时的点使用不同标记，且不参与推荐。

每个 baseline 使用稳定颜色和水平虚线，在两幅图的图例中注明名称及数值。只有验证和计时范围可比较的 baseline 才作为正式基准；不支持当前输入或缺测时，图中注明原因。待验证结果须与正式基准区分。版本使用点线，标注输入 ID、shape/dtype、设备和单位，避免用双纵轴混淆性能与耗时。

`metrics.json` 不要求 harness 采用固定 schema，但必须能还原报告和图，至少记录：

- 输入定义、完整 baseline/版本列表、每项状态及原因。
- 普通计时样本或其文件链接、warmup/repeat、平均值、verification 及阈值来源。
- 逻辑 FLOPs 公式及数值，按执行单元/精度划分的执行 FLOPs，计数来源与时间分母。
- 每个设备 kernel 的 launch 身份、次数、时间、计算和访存分项。
- DRAM/shared 的读、写、总字节及覆盖范围，各派生吞吐、峰值和利用率。
- 硬件、频率、理论峰值及来源，profile/源码/验证路径。

图和报告使用同一份汇总数据，不分别手填两套数字。缺失项保留 `null` 和原因；逻辑工作量未确定时性能图标注待补，不能改用不兼容指标填空。
