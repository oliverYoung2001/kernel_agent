# FlashKDA 来源与提炼边界

本套技能于 2026-11-19（Asia/Shanghai）整理。来源是本机 session 索引中名为 **FlashKDA** 的会话，thread ID：`01a1874b-a8bc-7222-940b-214e5b150654`。

会话早期分析 FlashKDA kernel，后续大量实验实际位于相邻 **LACP** 仓库。这两者不是同一个 Git 根。可迁移流程放在 `kernel_agent/skills/`，具体 Python case 放在 `kernel_agent/kernels/`；不复制 CUDA 实现、大型 trace、模型数据或完整性能档案。

`describe-gpu-kernel` 补充了显式 Python 输入/输出接口、可执行 reference 和每种 kernel 20 组 inputs 的交付要求。具体 KDA 模块参考实现、输入配方与来源指纹记录在 `kernels/kda_megakernel/README.md`。仅供 agent 使用的自检脚本及结果已移入 `tmp/`。

`kernel-verification` 按用户最新要求收敛为：每个优化版本对每种确定输入，与 reference 的所有输出做自定义阈值精度比较，返回逐输入 pass/fail。历史下游检查和同步诊断经验不再作为该 skill 的必需流程。

`iterate-kernel-optimization` 按用户最新要求采用最简流程：先从 reference 实现首版 megakernel，再按改进实现、使用 `kernel-profiling` 做 profiling 并充分预热/重复测量端到端平均耗时、分析结果的顺序迭代。

版本从 `v0` 开始，每轮新建递增的 `v{xxx}` 文件夹。`kernel-profiling` 使用 NCU、CUPTI、侵入式 timestamp 插桩；本轮实际 profiling 与端到端数据保留在对应版本目录，临时自检文件留在 `tmp/`。

来源路径用于追溯，不是运行依赖。迁移后六个 skill 自身及其模板仍可使用；复现历史实验才需要相应项目与原始产物。

## Conda / Spack 环境的具体来源

`prepare-environment` 按用户指定的 `~/yhy/llm/LACP/scripts/envs.sh` 重写，保留 `numactl@2.0.18`、`linear`、Conda 内 `nvidia/cu13` 路径、GCC 13.4.0 和激活顺序。本地 `envs.sh` 另外显式初始化 Conda/Spack，并兼容调用方的 nounset。

本目录的 `build.sh` 集中执行首次构建、激活检查并打印完成信息；`envs.sh` 专门用于 `source` 激活。

来源脚本只负责激活。构建说明另核对 `LACP/scripts/install.sh` 中的 `conda create -n linear python=3.12`，以及当前 `linear` 的 Python/Python 包元数据和 Spack 已安装 spec：Python 3.12.12、PyTorch 2.11.0+cu130、torchvision 0.26.0、torchaudio 2.11.0、CUDA nvcc/crt/runtime/cccl/nvvm 13.2.86。脚本不再包含工作负载契约或实验记录模板。

```text
0b669a161851a5ea1f6a3e2502981a1853b69081f98337871df0dfd7c2a8dfbc  LACP/scripts/envs.sh
6a496142ec324d3a44a42fb30d296f62a61f26820f1c349bf8d3263733616e0b  LACP/scripts/install.sh
```

## 原始会话定位

原始 JSONL 位于本机 `.codex/sessions/`。以下为读取时的文件与行号，timestamp 使用原始 UTC；会话仍可能追加内容。

| 文件（相对 sessions/） | 原始位置 | 提炼内容 |
|---|---|---|
| `2026/10/29/rollout-2026-10-29T18-54-05-01a1874b-a8bc-7222-940b-214e5b150654.jsonl` | line 10、145；2026-10-29 | per-warp/WG 分析；CUDA/Tensor、global/shared 分开；源码行号及细粒度依赖，不添加 false dependency |
| `2026/10/29/rollout-2026-10-29T22-22-20-01a1874b-a8bc-7222-940b-214e5b150654_01a1880a-52b4-78e0-9255-ffa06da17fce.jsonl` | line 679、989、1270；2026-10-29 | 符号 FLOPs/bytes、per block 汇总与并行配置，具体数值特化 |
| 同上 | line 1422；2026-10-30T15:39:35.781Z | 共享可复现输入、中间变量精度、顺序 reference、实现→验证/采集→分析循环、每版产物和复现脚本 |
| 同上 | line 8932、8992、9012、9078、9105、9132、9168；2026-11-01 至 03 | 版本记录、验证与 baseline 入口、一次命令覆盖范围、英文代码注释、误差指标与 case/sample 语义 |
| 同上 | line 9274、9391、9456、9636；2026-11-03 | 宏分支、grid/warp/shared、persistent task 分区、调度及依赖 |
| 同上 | line 10548、10603、10700、10751；2026-11-04 | baseline 调用/验证链、bitwise 条件、精度路径/原始 FLOPs、每 kernel 与总量报告 |
| `2026/11/05/rollout-2026-11-05T02-16-36-01a1874b-a8bc-7222-940b-214e5b150654_01a1a7c6-f3e0-78c1-a8f8-5976c6aab15b.jsonl` | line 34672；2026-11-10T16:41:52.550Z | 汇总 230 个版本的成功、失败与诊断经验为可复用 skill |

没有将原任务“某项 utilization 达到 90%”作为通用停止规则。新任务需要自己定义目标和 denominator；也没有继承另一个 session 的长任务授权或活跃目标。

## 读取的维护资料

下列路径均相对本机 `FlashKDA/skills/`，内容在整理时已实际读取。长会话后续更正通过这些维护资料交叉核实，因此本次提炼不限于旧的 v1–v230 数值结论。

| 来源 | 进入本套 skill 的经验 |
|---|---|
| `optimize-hopper-gemm/SKILL.md` | 实际 checkout 与实现家族；独立验收层；每轮假设和可审计产物 |
| `optimize-hopper-gemm/references/lacp-runbook.md` | 冻结 request/source/launcher、真实加载 binary、完整输出与下游验证、fresh 与 replay 区分 |
| `optimize-hopper-gemm/references/measurement.md` | logical/executed FLOPs、memory 口径、指标分母、配对/A/A、clock scope、三种插桩对照、局部上限限制 |
| `optimize-hopper-gemm/references/experiment-record.md` | 逐轮身份、假设、验证、计时、profile、决策与失败记录 |
| `optimize-hopper-gemm/references/qualification-update.md` | 后续 barrier 生命周期缺陷：历史数值通过不是同步正确证明，跨 task init/invalidate 必须独立审查 |
| `optimize-hopper-gemm/references/shared-ticket-handoff.md` | 发布共享 ticket 与全部 reader 退休是不同条件；终止和角色 handoff 需要检查覆盖前的 reader 完成 |
| `optimize-hopper-gemm/references/root-codegen-learning.md` | 静态 spill/unroll 与动态流量不同；设备代码身份审查；零 spill 未必更快；不编辑运行中的 launcher |
| `gpu-sass-simulation/SKILL.md` | capture/replay 完整性、模型事件边界、独立留出与反例、cycle/time error 分开、依赖覆盖不等于时间精度 |

另读取 LACP 中的实际验证实现：

- `lacp/la_ops/kda/megakernel_agent/metrics.py`：有限性、绝对/相对误差、scaled_max、rmse_ratio、分块归约与序列曲线。
- `tests/test_glm5_3_flash_megakernel.py`：case ID/RNG 配方、同输入 baseline、逐中间变量检查、特殊 gate/state 与完整覆盖。

这些公式和方法已在本套文件中重新表述。特定实验的 0.01 门限、seed 42、23 tensors × 3 cases、shape、设备容量、版本名和结果值不作为新任务要求。

## 资料快照指纹

以下 SHA256 对应本次读取时的资料，不宣称原仓库后续保持不变。原始 session 可能继续追加，未对它声明固定 hash。

```text
0f9b847a217e52a200f22f9bb1d571f5eacc61c33c1bf0fb1b07683885213ef3  optimize-hopper-gemm/SKILL.md
fdad0948514bbd7cc820865f2861b708dbcbe2655f7db411fa9fb86f5554d838  optimize-hopper-gemm/references/measurement.md
05e880fd6211a38291202266ebc11f91ad6f338c1cdc07612f573d375ce46212  optimize-hopper-gemm/references/lacp-runbook.md
c9aecd73d69fb5b58566e26960a532c4b9fd235f376ed5fe7813459c25493bfa  optimize-hopper-gemm/references/experiment-record.md
0670f95836a85bd338ac08b6b3609ba72dbd1990101738e3aeaf5c253610d240  optimize-hopper-gemm/references/qualification-update.md
eeb125e6ea501c86a16fe0647064027f3cbe6297067fc1e70b51f3d726da26ee  optimize-hopper-gemm/references/shared-ticket-handoff.md
5831173b1634734e9fc7ea5d657b13b0e3bae3b8a1e2a0837e294aae73c5b5a5  optimize-hopper-gemm/references/root-codegen-learning.md
b8dd84ac63ddf44f5c4004766ff2c7e65103e8b29aeec671c519e930db6fa0f7  gpu-sass-simulation/SKILL.md
2b4c57c3669ac0c0825d5464a47dda7a981a301c4976ed2d35680b99e0cbde2f  LACP/lacp/la_ops/kda/megakernel_agent/metrics.py
b21a8d187c5fca900919f661c0ad306bb9d3d8e9da5a7680577bd01e315e6509  LACP/tests/test_glm5_3_flash_megakernel.py
```

技能组织与格式采用本机 `skill-creator` 指南：精简入口、按需读取 reference、保留真实约束并避免将单个历史案例变为通用规定。没有引入 agent 专属工具调用、模型配置或固定安装目录。

## 实验报告的用户约定

报告按当前用户要求保存在目标 kernel 的 `reports/` 下，覆盖全部 baseline、版本和输入。指标包括完整调用平均时间、Tensor/CUDA Core 计算量与吞吐/理论利用率、DRAM/shared 读写流量与吞吐/理论利用率，并展开各设备 kernel。每个输入的图片包含性能、耗时两幅版本对比图，baseline 使用水平虚线。该要求定义交付格式，不代表当前已经完成这些测量。
