---
name: kernel-verification
description: Compare every output of an optimization version against a reference on each fixed kernel input, apply configurable numerical thresholds, and return one pass or fail per input.
---

# Kernel 精度 verification

在 agent 每次迭代产生一个优化版本后，对 kernel 定义的每一种确定输入运行该版本与 reference，比较所有输出，返回逐输入的 `pass` / `fail`。

## 输入

- 当前优化版本的 Python function 和 reference function，两者使用相同的显式输入/输出接口。
- kernel 已定义的全部输入 case，暂定每种 kernel 20 组；复用输入生成器，不在 verifier 内重新定义 shape、dtype 或分布。
- reference 的全部输出名称，包括 output 和 inout 参数；例如 KDA 的 `out` 与 `final_state`。
- 自定义精度阈值，可统一设置或按输出分别设置，例如 atol/rtol、scaled_max、rmse_ratio。阈值在本轮比较前确定，多个 gate 同时设置时需要全部满足。

## 对每一种输入执行

1. 将该 case 具体化一次，得到确定的 tensor 值和 scalar 参数。
2. 为 reference 与当前优化版本准备数值相同、存储独立的实参。恢复相同初始状态和 inout 初值，分别预分配输出。
3. 分别调用 reference 和当前版本，等待计算完成；不能让 reference 的写入成为当前版本的输入。
4. 比较 reference 声明的**所有输出、所有元素**。按接口核对 shape/dtype，检查非有限值，并应用本轮自定义精度阈值。
5. 所有输出均满足要求时，该输入返回 `pass`；任一输出超限、缺失、结构错误、非有限或运行异常时，返回 `fail`。
6. 继续验证其余输入，不在首个 fail 后结束整个输入集合。

执行流程和阈值公式见 [verification 实现方法](references/verification-implementation.md)。

## 返回值

verification 的主要结果就是一组输入 ID 到 `pass` / `fail` 的映射：

```json
{
  "kda_01": "pass",
  "kda_02": "fail"
}
```

这是格式示例，不是已运行结果。真实执行需要为全部输入返回一个状态；20 组输入就应有 20 个互不重复的输入 ID。只有全部为 pass，当前版本才整体通过精度验证。结果格式也见 [JSON 示例](templates/verification-result.json)。

没有运行某组输入时，不能将它填成 pass 或宣称验证完成。实际运行中的异常应记为 fail；资源不足导致整个验证无法完成时，保留实际进度并明确未完成。

误差数值、失败输出名称、异常信息等辅助诊断按需放到 `tmp/`，不扩充主要返回值。仅供 agent 自检的脚本、日志和临时报告也放 `tmp/`。生成代码的注释使用英文。
