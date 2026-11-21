# KDA megakernel forward case

## 优化注意点

设计和评估优化时，必须覆盖逐 token 衰减因子 $\alpha$ 任意接近 1 的弱衰减情形。令 $X$ 为输入 RMSNorm 后的 hidden states，则

$$
\alpha =
\exp\!\left(LB \cdot \operatorname{sigmoid}\!\left(
\exp(A_{\log}) \cdot (X W_{\alpha}^{\downarrow} W_{\alpha}^{\uparrow} + \mathrm{dt\_bias})
\right)\right)
\in (e^{LB},1)^{B \times S \times H_v \times D_k}.
$$

当前实现取 $LB=-5$，代码中的 `alpha_log` 是上式外层指数的输入，即 $\log\alpha$。当前模型 case 中 $H_v=H_{qk}$。

弱衰减时，较早 token 写入的状态可能在很长的序列内仍有贡献。不得依据衰减把历史状态视为可忽略，从而截断递推、重置块初态或跳过跨块状态传递。也不得因为 $\alpha$ 接近 1 就把它替换为 1，逐 token 的差别可能在长序列上累积。修改递推或块间状态传递时，补充长序列、$\alpha$ 接近 1 的验证，并按既有门限比较 `out` 和 `final_state`。

## 实验记录

[experiment0 报告](experiment0/reports/report.md) · [实验选择与复现命令](tools/README.md) · [Profiling 与报告重建](profiling/README.md)

[experiment1 完整性能报告（十轮优化、独立复测、逐输入与逐 launch 数据）](experiment1/reports/report.md)
