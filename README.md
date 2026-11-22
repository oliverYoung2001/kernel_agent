# kernel_agent

本仓库提供六个 GPU kernel 工作流 skill，以及 [KDA megakernel](kernels/kda_megakernel/README.md) 和 [分布式 KDA module](kernels/kda_module_dist/README.md) 两种 workload。分布式版本提供 TP、Ulysses SP、KCP 三个 baseline、8/16 卡的 20 个输入配方及任意合法并行度的函数接口。下面说明如何在 Codex 中加载和使用；技能的详细规范见 [skills/README.md](skills/README.md)。

## 让 Codex 发现这些 skill

Codex CLI 和 IDE 扩展支持通过 `/skills` 查看、选择技能，或在提示词中输入 `$skill-name` 显式调用，也可以根据任务自动匹配技能。仓库级发现目录为 `.agents/skills/`，用户级目录为 `~/.agents/skills/`，支持链接到实际 skill 文件夹；本仓库的 `skills/` 是源码目录，需要链接后才能按上述方式发现。参见 [Codex 官方 skills 文档](https://developers.openai.com/codex/skills/)。

在本仓库根目录执行以下命令，将全部六个 skill 链接为仓库级技能：

```bash
skill_target="$PWD/.agents/skills"
mkdir -p "$skill_target"
for skill_dir in "$PWD"/skills/*; do
    [ -f "$skill_dir/SKILL.md" ] || continue
    skill_name="${skill_dir##*/}"
    if [ -e "$skill_target/$skill_name" ] || [ -L "$skill_target/$skill_name" ]; then
        printf 'Already exists, skipped: %s\n' "$skill_target/$skill_name"
        continue
    fi
    ln -s "$skill_dir" "$skill_target/$skill_name"
done
```

需要在其他仓库也使用时，将第一行改为 `skill_target="$HOME/.agents/skills"`，仍在本仓库根目录执行整段命令。仓库级与用户级选择一种即可，避免同名技能重复出现。已有目标会跳过，请自行核对是否指向本仓库。

链接后，在 Codex 中输入 `/skills` 检查这六个名称。若没有出现，重启 Codex 会话后再检查。链接随源码更新，无需重新复制；移动仓库后需要更新链接。

也可以不安装，直接在 Codex 中要求读取文件：

```text
阅读 skills/iterate-kernel-optimization/SKILL.md，并按其中的规范优化
kernels/kda_megakernel；同时读取该 skill 引用的必要文件。
```

下面的 `$...` 示例都是发给 Codex 的提示词，不是终端命令。在本仓库启动 Codex，或在 IDE 中打开本仓库后使用。

## 六个 skill 的用途

| Skill | 负责的工作 | 主要产物 |
|---|---|---|
| [prepare-environment](skills/prepare-environment/SKILL.md) | 构建和激活 LACP 配置的 Conda/Spack 环境 | `build.sh` 构建；`envs.sh` 激活 |
| [describe-gpu-kernel](skills/describe-gpu-kernel/SKILL.md) | 定义 Python 接口、reference 和 20 组不同输入，描述计算与依赖 | `kernels/<name>/` 中的 case 文件与说明 |
| [kernel-verification](skills/kernel-verification/SKILL.md) | 各实现与首个 baseline 使用相同确定输入，对比所有输出 | 全部输入 ID 到 `pass` / `fail` 的映射 |
| [iterate-kernel-optimization](skills/iterate-kernel-optimization/SKILL.md) | 至少 10 轮实现、改进、测量与分析 | 从 `v0/` 起递增的版本目录 |
| [kernel-profiling](skills/kernel-profiling/SKILL.md) | NCU、CUPTI、侵入式 timestamp profiling 与独立端到端测试 | 对应版本内的 `profiling/`、`end_to_end.json` |
| [report-kernel-experiments](skills/report-kernel-experiments/SKILL.md) | 汇总所有 baseline、版本及输入，生成报告和版本对比图 | 当前实验的 `reports/` 下的报告、汇总数据与图片 |

## 分别调用

### 环境构建与激活

首次构建，已有 Conda 和 Spack 安装时：

```text
$prepare-environment
请使用 skills/prepare-environment/build.sh 构建环境，并检查构建结果。
```

已有环境，只需激活：

```text
$prepare-environment
环境已经构建，请使用 skills/prepare-environment/envs.sh 激活，
随后运行 kernel 命令时在同一个 shell 中先 source 这个脚本。
```

也可以自己在终端执行：

```bash
bash skills/prepare-environment/build.sh
source skills/prepare-environment/envs.sh
```

`build.sh` 以 LACP 的配置构建 `linear`、CUDA 编译工具链和 Spack numactl/GCC，并从源码按计算节点可见 GPU 的计算能力编译安装 FlashKDA，已有 `linear` 时不覆盖。已有环境可跳过构建，按 [prepare-environment 的说明](skills/prepare-environment/SKILL.md#安装-flashkda) 单独补装 FlashKDA。构建脚本不能修改调用它的父 shell；需要用 `source envs.sh` 激活。Codex 使用新的 shell 执行后续命令时，应在该 shell 中重新 source。kernel 自身的 FLA 等其余依赖按 [case 说明](kernels/kda_megakernel/README.md) 另行准备。

### 定义和描述 kernel

```text
$describe-gpu-kernel
为新 kernel 创建 kernels/<kernel-name>/，提供显式 Python function 接口，
所有输入和输出缓冲区均写在函数参数中。
实现可执行 reference，定义 20 组可复现输入，覆盖不同 shape、dtype 和数值分布。
说明各阶段计算、访存和依赖，并给出运行方式。
```

将 `<kernel-name>` 替换为目标名称，并提供数学定义或已有源码。已有 KDA case 可直接指定 `kernels/kda_megakernel/`，让 Codex 读取或补充描述，无需重新生成 reference 和输入集。

### 精度验证

```text
$kernel-verification
验证 kernels/kda_megakernel/experiment0/v0/ 中的实现。
对全部 20 组输入，通过 run.py --mode verify 与第一个 baseline 比较 out 和 final_state 的所有元素。
本次使用 scaled_max <= 0.01 且 rmse_ratio <= 0.01，两个条件同时满足才通过。
返回每个输入 ID 对应的 pass/fail，诊断文件放在当前实验的 tmp/。
```

`v0` 和上述阈值只是调用示例，应换成要检查的版本及本次要求的阈值。ground truth 与候选版本使用数值一致、存储独立的输入，分别准备输出；任一输出失败则该输入为 fail，不在首个失败后停止其余输入。

### 迭代优化

```text
$iterate-kernel-optimization
优化 kernels/kda_megakernel，使用现有 reference、20 组输入及 baseline.py。
新实验新建 experiment{N}/ 并从 v0 开始；恢复已有实验时从其当前版本继续。
使用 kernel-verification 验证所有输出，本次 scaled_max 和 rmse_ratio 均不得超过 0.01。
通过后使用 kernel-profiling 做 profiling，并独立测量普通完整函数端到端平均时间。
warmup=100、repeat=1000 起步，按稳定性调整并记录实际次数。
每轮分析结果后继续改进，至少完成 10 轮（v0 计为第 1 轮）后汇报。
```

迭代顺序为：实现/改进 → 精度验证 → profiling 与端到端测试 → 分析结果。首版从 `v0` 开始，至少完成 10 轮，即从零开始至少保留 `v0`–`v9` 的源码和实际数据，不覆盖旧版。继续已有任务时按已完成轮次补足，重复补测不计为新轮次；用户可指定更多轮、目标设备、阈值和测量次数。

### 单独 profiling 和计时

```text
$kernel-profiling
分析 kernels/kda_megakernel/experiment0/v0/ 的实现，对定义的全部输入分别采集。
使用 NCU 分析 Tensor/CUDA Core 计算及 DRAM/shared 读写；
需要执行时间线或阶段时间时，可使用 CUPTI 或侵入式 timestamp。
独立运行普通完整函数计时，充分 warmup 后重复测量并取算术平均值。
原始工具结果保存到 v0/profiling/，计时样本和平均值保存到 v0/end_to_end.json。
```

将版本替换为实际目标；工具按问题选择，不要求每轮全部使用。完整调用计时包含实现内部必要的转换和准备，profiling/插桩结果不混入正式端到端样本。

### 生成实验报告

```text
$report-kernel-experiments
为 kernels/kda_megakernel 生成报告，保存到当前实验的 reports/ 文件夹。
覆盖 baseline_fla、baseline_flashkda 和全部已有版本，按全部 20 组输入分别比较。
列出端到端平均耗时；Tensor/CUDA Core 的 TFLOP、TFLOP/s 和理论峰值利用率；
DRAM/shared load/store 的 traffic、throughput 和理论带宽利用率。
同时展开各设备 kernel 明细。
每个输入生成一张性能/耗时两幅子图的图片，横轴为版本号，baseline 使用水平虚线。
本次仅整理已有数据；缺测或不支持的项目注明原因，不启动新的 GPU 实验。
```

如需补齐测量，将最后一句改为“补测缺失项，按 kernel-profiling 保存原始结果后生成报告”，并指定可使用的 GPU/执行资源和测量预算。

## 组合使用

新 kernel 的工作顺序为：`prepare-environment` → `describe-gpu-kernel` → `iterate-kernel-optimization`（内部使用 verification 和 tools）→ `report-kernel-experiments`。已有 reference 和环境时，可以直接从优化、验证、profiling 或报告开始。

例如在一次 Codex 任务中完成优化与报告：

```text
$iterate-kernel-optimization $report-kernel-experiments
目标为 kernels/kda_megakernel，使用已有环境、reference、20 组输入和两个 baseline。
每轮用 kernel-verification 检查全部输出，本次 scaled_max 和 rmse_ratio 均 <= 0.01。
从 v0 开始实现首版，已有版本则继续递增，至少完成 10 轮优化（含 v0）。
每轮使用 kernel-profiling，充分 warmup 后重复测量完整调用平均时间并分析 profiling。
最后在当前实验的 reports/ 下生成全部 baseline/版本的指标表和性能/耗时对比图。
允许补测缺失数据；设备和资源使用本次任务指定的可用 GPU。
```

使用时提供目标 kernel、版本/轮数、GPU 资源、精度阈值及是否允许补测。Codex 应从仓库读取已有定义和数据，完成后报告实际执行范围及未完成项。

## 文件保存位置

```text
kernels/<kernel-name>/
├── reference.py
├── baseline.py
├── run.py              # Shared reference/baseline/version runner
├── verification/       # Shared numerical checks for all experiments
├── profiling/          # Shared CUPTI and NCU collectors and trace analysis
├── cases/
│   ├── inputs.py
│   ├── generate_cases.py
│   ├── cases.json
│   └── cases_0.json
└── experiment{N}/
    ├── v0/
    │   ├── <implementation files>
    │   ├── profiling/
    │   └── end_to_end.json
    ├── v1/
    │   └── ...
    ├── baselines/<baseline-name>/
    │   ├── profiling/
    │   └── end_to_end.json
    └── reports/
        ├── report.md
        ├── metrics.json
        └── figures/<input-id>_comparison.png
```

以上是正式产物的保存约定，不表示当前已经存在所有版本、测量或报告。仅供 agent 使用的自检脚本、临时日志和检查产物放在当前实验的 `tmp/`，该目录由 `.gitignore` 排除。
