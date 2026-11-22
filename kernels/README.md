# Kernel cases

每个目录定义一种可替换的计算单元，提供显式输入/输出的 Python 函数、可执行 reference 和暂定 20 组输入配方。reference 允许由多个 kernel 拼接完成。

| Case | 范围 | 接口与输入 |
|---|---|---|
| [kda_megakernel](kda_megakernel/README.md) | 输入 RMSNorm、QKV 与 gate 投影、因果卷积、KDA 递推、门控 RMSNorm 和输出残差 | [reference.py](kda_megakernel/reference.py) · [两个 baseline](kda_megakernel/baseline.py) · [20 组 inputs](kda_megakernel/cases/cases.json) |
| [kda_module_dist](kda_module_dist/README.md) | 同一完整模块的分布式版本；token 连续切分；TP、Ulysses SP、KCP；并行度由 process group 决定 | [reference.py](kda_module_dist/reference.py) · [三个 baseline](kda_module_dist/baseline.py) · [20 组 inputs](kda_module_dist/cases/cases.json) |

新增 case 的接口和输入要求见 [describe-gpu-kernel](../skills/describe-gpu-kernel/SKILL.md)。

# 性能优化指导

你作为kernel agent写初始版本和优化的时候，需要实现为一个cuda megakernel，内部你可以使用cutlass，在'third_party/cutlass'中。
