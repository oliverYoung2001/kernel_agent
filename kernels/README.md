# Kernel cases

每个目录定义一种可替换的计算单元，提供显式输入/输出的 Python 函数、可执行 reference 和暂定 20 组输入配方。reference 允许由多个 kernel 拼接完成。

| Case | 范围 | 接口与输入 |
|---|---|---|
| [kda_megakernel](kda_megakernel/README.md) | 包含投影、卷积、KDA recurrence、门控 RMSNorm 和残差的完整 forward 模块 | [reference.py](kda_megakernel/reference.py) · [两个 baseline](kda_megakernel/baseline.py) · [20 组 inputs](kda_megakernel/cases.json) |

新增 case 的接口和输入要求见 [describe-gpu-kernel](../skills/describe-gpu-kernel/SKILL.md)。
