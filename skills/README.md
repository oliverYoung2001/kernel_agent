# 可迁移的 GPU kernel 工作流 skills

这套技能将 **FlashKDA session** 中的环境准备、verification、kernel 描述、优化循环、外部工具和阶段报告方法拆成六个可独立使用的 skill。适用于不同模型、算子、agent 和执行 harness；内容包括 Markdown 指令、环境脚本、记录模板和验收规则，不依赖某个 SDK 或固定 runner。

整理依据与具体经验的来源见 [SOURCES.md](SOURCES.md)。环境 skill 按 LACP `scripts/envs.sh` 保留具体 Conda/Spack 配置；其他 skill 中的历史实现、GPU 参数、模型配置和测量值按当前任务替换。

## 选择哪个 skill

| Skill | 什么时候用 | 主要交付 |
|---|---|---|
| [prepare-environment](prepare-environment/SKILL.md) | 构建和激活 LACP 使用的 Conda/Spack 环境 | 一键构建 [build.sh](prepare-environment/build.sh)，激活 [envs.sh](prepare-environment/envs.sh) |
| [kernel-verification](kernel-verification/SKILL.md) | 每轮检查优化版本在全部输入上的输出精度 | 与 reference 比较所有输出，按自定义门限逐输入返回 pass/fail |
| [describe-gpu-kernel](describe-gpu-kernel/SKILL.md) | 定义 kernel case、解释并行结构与工作量 | 显式 Python 输入/输出接口、reference、20 组输入，以及实现结构/依赖描述 |
| [iterate-kernel-optimization](iterate-kernel-optimization/SKILL.md) | 从 `v0` 开始，每轮新建 `v{xxx}` 改进实现 | 改进实现 → profiling/端到端平均耗时测试 → 分析并决定下一轮改进 |
| [use-kernel-tools](use-kernel-tools/SKILL.md) | 使用 NCU、CUPTI、侵入式 timestamp 插桩做 profiling | profiling 与端到端数据保留在对应的 `v{xxx}` 文件夹 |
| [report-kernel-experiments](report-kernel-experiments/SKILL.md) | 汇总全部 baseline 和版本的实验结果 | kernel 的 `reports/` 下保存报告、计算/访存指标和版本对比图；baseline 用虚线 |

每个 skill 都有 YAML `name`/`description` 和 Markdown 正文。详细模板放在各 skill 的 `references/` 或 `templates/`，按需读取。优化循环会使用 `kernel-verification` 检查精度，并使用 `use-kernel-tools` 做 profiling。

## 如何在不同模型或 harness 中使用

**支持 skill 加载的环境**：将完整 skill 文件夹放入该环境支持的发现目录，按其规则通过名称选择或自动匹配。发现目录、调用语法和可用工具由你的环境决定；仓库中的 `skills/` 是分发位置，不保证任何客户端会自动扫描。

**普通聊天模型或自定义 agent**：让模型读取需要的 `SKILL.md`，再按正文提示读取相对路径的参考文件；不能访问文件时，把对应正文和需要的模板提供给它。不需要实现任何专用 tool API。

**迁移到另一个项目**：复制完整 skill 文件夹，保持内部相对路径。不要只复制入口而遗漏模板。选择一个 skill 就能独立使用；整套复制可保留本 README 和来源说明。

**接入自己的 harness**：将项目现有的 build、execute、verify、measure、profile 入口对应到技能要求的职责。使用现有 CLI、函数、容器或批处理脚本均可，不要求专用 SDK、JSON 协议或调度器。遵循 kernel 的显式函数接口，以及下文约定的版本数据和报告目录；JSON 模板仅是可采用的结果字段示例。

`prepare-environment` 提供具体的环境构建命令与激活脚本，其余技能提供工作方法和模板。实际 kernel、模型输入、reference、profiler 和 simulator 由使用项目提供。加载技能不等于已经执行 verification 或获得性能结果。

## 给任务提供哪些信息

尽量给出：源码根与目标入口，算子/模块范围，shape/stride/layout，输入/累加/输出精度，状态语义，reference 和精度门限，目标设备，完整计时边界，允许的算法变换，以及目标/预算/停止条件。

已有仓库包含的信息让执行者自行读取。缺少的信息按当前代码和任务核实；历史 seed 42、0.01 门限、69 项检查、某个 tile/width 或 H20 参数都不是通用默认值。

## 使用示例

准备环境：

```text
阅读 skills/prepare-environment/SKILL.md。
运行其中的 build.sh 构建 Conda linear 与 Spack numactl/GCC 环境；
已有环境则直接 source skills/prepare-environment/envs.sh。
```

在本仓库根目录首次构建，随后激活；已有环境时只执行 `source`：

```bash
bash skills/prepare-environment/build.sh
source skills/prepare-environment/envs.sh
```

实现 verification：

```text
阅读 skills/kernel-verification/SKILL.md 及其验证实现规范。
让当前优化版本与 reference 接收同一组确定输入和初始状态，
比较 reference 的所有输出，并使用我设置的精度阈值。
对 kernel 的全部 20 组输入逐一执行，返回输入 ID 到 pass/fail 的映射。
```

描述 kernel：

```text
阅读 skills/describe-gpu-kernel/SKILL.md。
为每种 kernel 在 kernels/ 下提供显式 Python function、可执行 reference
和 20 组可复现输入，输出缓冲区也写在 function 参数中。
解析当前选中的模板/宏路径，给出 grid、warp/WG 角色和 task 划分。
用符号公式和具体数值分别统计计算与访存，链接代码位置。
按对象、slot 和 generation 解释依赖，不从绘图位置添加同步关系。
```

具体实例见 [KDA megakernel case](../kernels/kda_megakernel/README.md)：完整模块 reference、20 组 shape/dtype/分布配方与运行入口。生成方法见 [Kernel case 规范](describe-gpu-kernel/references/kernel-case.md)。

持续优化并总结：

```text
阅读 skills/iterate-kernel-optimization/SKILL.md，
先依据 reference 实现 megakernel，保存到 v0/。
之后每轮新建 v1/、v2/ 等版本目录，在新目录改进实现，
再使用 use-kernel-tools 的 NCU、CUPTI 或 timestamp 插桩做 profiling，
并在充分 warmup 后重复足够次数，计算端到端耗时的算术平均值。
本轮代码、profiling 和端到端数据都保存在对应的版本文件夹。
最后分析 profiling 与性能结果，用来决定下一轮改进。
每次改动后用 kernel-verification 检查全部输入的精度。
```

生成实验报告：

```text
阅读 skills/report-kernel-experiments/SKILL.md。
将 kda_megakernel 的报告保存到 kernels/kda_megakernel/reports/。
覆盖 baseline_fla、baseline_flashkda 和全部 v{xxx}，对全部输入分别列出：
完整调用平均时间，Tensor/CUDA Core 的 TFLOP、TFLOP/s 和理论峰值利用率，
DRAM/shared 的 load/store traffic、throughput 和理论带宽利用率。
同时展开各设备 kernel 的指标。
每个输入生成一张含性能和耗时两幅子图的图片，横轴为版本号，baseline 为水平虚线。
此任务只整理已有证据，不新增 GPU 实验。
```

## 组合工作流

新任务先构建或激活环境并定义 kernel 的 reference、接口和输入集合；随后实现首版 megakernel，按“改进实现 → profiling/端到端性能测试 → 分析结果”循环优化。

已有完整验证的项目可以直接进入优化；只解释代码或汇总报告也可以单独使用相应技能。不要为了形式完整而运行无关工具或重建整个 harness。

跨阶段关联始终依据 workload、解析后的配置、源码和实际 binary。以下证据保持独立：

- 准备/构建/启动完成。
- 完整数值及下游检查完成。
- 内存/同步诊断的实际覆盖。
- 普通完整调用的性能结果。
- profiling、插桩和模拟结果。

这种区分保留了 FlashKDA 的重要教训：历史数值通过可能后来发现同步协议缺陷；减少流量或 spill 不保证变快；接近局部模型不保证达到全局性能上限。

## 目录与维护

```text
skills/
├── README.md
├── SOURCES.md
├── prepare-environment/
│   ├── SKILL.md
│   ├── build.sh
│   └── envs.sh
├── kernel-verification/
│   ├── SKILL.md
│   ├── references/verification-implementation.md
│   └── templates/verification-result.json
├── describe-gpu-kernel/
│   ├── SKILL.md
│   └── references/
│       ├── kernel-case.md
│       └── kernel-description.md
├── iterate-kernel-optimization/
│   ├── SKILL.md
│   └── references/iteration-record.md
├── use-kernel-tools/
│   ├── SKILL.md
│   └── references/tool-evidence.md
└── report-kernel-experiments/
    ├── SKILL.md
    ├── references/metrics-and-plots.md
    └── templates/experiment-report.md
```

维护时把会改变执行者决策的通用规则留在入口，把较长规范留在引用文件；新增经验附上适用条件和来源。不要把一次失败固化为所有 GPU/模型的禁令，也不要把历史参数补进通用默认值。生成的 shell、Python 和 CUDA 代码注释使用英文，报告语言按任务要求。

格式、引用和模板有效只是这套说明文件的校验；真正使用后仍需执行当前 kernel 的测试与测量。

仅供 agent 使用的自检脚本、临时日志和检查报告统一放在 `tmp/`；该目录已由 `.gitignore` 排除。

优化的正式产物保留在 `kernels/<kernel-name>/v0/`、`v1/` 等目录：每轮源码、`profiling/` 下的工具数据与 `end_to_end.json` 必须属于同一版本，不能覆盖旧版或混入其他版本结果。

报告及其汇总数据、图片放在 `kernels/<kernel-name>/reports/`，原始测量数据仍留在对应版本。新采集的 baseline 数据放在 `kernels/<kernel-name>/baselines/<baseline-name>/`，不占用 `v{xxx}` 编号。
