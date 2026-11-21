---
name: kernel-verification
description: Verify every kernel implementation against the first declared baseline on identical inputs through run.py --mode verify, comparing all outputs under configurable numerical thresholds.
---

# Kernel 精度 verification

## 原理

每种 kernel 按声明顺序选择第一个 baseline 作为 ground truth。只有一个 baseline 时就使用该实现。同一输入上，其他所有 baseline、reference 和当前实验的所有版本都分别与 ground truth 比较。ground truth 负责生成期望输出，记录其身份和执行状态；候选版本的排名或命令行筛选顺序不影响 ground truth 的选择。

## verification代码入口

统一通过 kernel 根目录的 `run.py --mode verify` 执行。默认枚举当前实验的全部待验证实现，运行全部固定 case 和 kernel 定义的专项精度检查。人工复查时可指定实现和 case。优化采集、最终复测和报告补测也调用该模式。

以 KDA 为例，从仓库根目录执行：

```bash
export KDA_EXPERIMENT=experiment1
python -m kernels.kda_megakernel.run --mode verify --list
python -m kernels.kda_megakernel.run --mode verify
python -m kernels.kda_megakernel.run --mode verify --impl v0 --case weak-packed
```

## verification代码生成

每个 kernel 的 `run.py` 只负责解析参数和分派功能。`kernels/{kernel_name}/verification/` 负责选择 ground truth、枚举待测实现、准备独立输入、运行标准和专项 case、比较全部输出、判定和保存结果。复用 kernel 的 baseline、reference、版本实现和输入生成器。

1. 对每个 case 具体化一次输入，保留标量、权重和 inout 初值。给 ground truth 和每个候选提供数值一致、存储独立的参数及输出缓冲区。
2. 先执行 ground truth 并等待完成，保存全部输出。对每个候选执行同一 case，按接口检查所有 output 和 inout 终值的 shape、dtype、有限性，以及每个元素参与的误差指标。
3. 使用运行前指定的阈值。一个 case 的所有输出和门限均通过时该候选才通过。任何实现执行或比较异常都记录原因，继续其余输入。ground truth 执行失败时，该输入的候选不能通过。
4. 按实现保存输入 ID 到 `pass` / `fail` 的结果，记录 ground truth、阈值和源码身份。专项检查标明 case 名。正式记录保存在对应实验的实现目录，临时诊断写入该实验的 `tmp/`。更换 ground truth 后重新验证，保留历史结果及其对照身份。

阈值公式与遍历流程见 [verification 实现方法](references/verification-implementation.md)。逐输入结果示例见 [JSON 示例](templates/verification-result.json)。生成代码的注释使用英文。
