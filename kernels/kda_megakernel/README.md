# KDA megakernel forward case

这个 case 定义一个完整 KDA 模块的可替换 Python 接口，并提供可执行的多算子 PyTorch reference 和 **20 组输入配方**。计算范围来自 FlashKDA session 中的 LACP GLM-5.3-Flash forward 任务：

`QKV projection → causal depthwise conv + SiLU → Q/K L2Norm → KDA recurrence → gated RMSNorm → output projection + residual`。

alpha_log、beta 和输出 gate 已作为输入准备好，生成它们的投影不在此范围内。forward 无反向和卷积历史状态，序列左侧使用零填充。

## 接口

入口为 [reference.py](reference.py) 中的 `reference`：

```python
def reference(
    inpx, in_proj_qkv_w, conv1d_weight, alpha_log, beta, gate,
    norm_weight, o_proj, initial_state, residual, out, final_state,
    *, num_heads_qk, num_heads_v, head_dim_qk, head_dim_v,
    l2_eps=1e-6, norm_eps=1e-5, scale=None,
) -> None:
    ...
```

记 `B=batch, S=sequence, M=hidden, Hqk=num_heads_qk, Hv=num_heads_v, Dk=head_dim_qk, Dv=head_dim_v, Kc=conv_width`，`C=2*Hqk*Dk+Hv*Dv`。`act` 为本组输入的 FP16、BF16 或 FP32。

| 参数 | 方向 | Shape / 类型 | 语义 |
|---|---|---|---|
| inpx | input | `[B,S,M]`, act | 输入 hidden states |
| in_proj_qkv_w | input | `[M,C]`, act | Q/K/V 投影权重，依次打包 |
| conv1d_weight | input | `[C,1,Kc]`, act | 无 bias 的逐通道因果卷积；最后一个权重对应当前 token |
| alpha_log | input | `[B,S,Hv,Dk]`, FP32 | 自然对数衰减，合法 case 中非正；每个 key channel 独立衰减 |
| beta | input | `[B,S,Hv]`, act | 已激活的 update gate，范围 `[0,1]` |
| gate | input | `[B,S,Hv,Dv]`, act | 输出 gate 的 logits，由 reference 内 sigmoid |
| norm_weight | input | `[Dv]`, act | 每个 value head 共享的 RMSNorm weight |
| o_proj | input | `[Hv*Dv,M]`, act | 输出投影权重 |
| initial_state | input | `[B,Hv,Dk,Dv]`, FP32 | key-major/value-minor 的初始 recurrent state |
| residual | input | `[B,S,M]`, act | 输出投影之后加上的残差 |
| out | output | `[B,S,M]`, act | 调用方预分配，完整写回最终输出 |
| final_state | output | `[B,Hv,Dk,Dv]`, FP32 | 调用方预分配，完整写回最后状态 |
| num_heads_qk / num_heads_v | input | positive int | `Hv % Hqk == 0`，每个 Q/K head 对应连续一组 value heads |
| head_dim_qk / head_dim_v | input | positive int | key/value head dimensions |
| l2_eps / norm_eps | input | positive finite float | L2Norm/RMSNorm epsilon |
| scale | input | finite float 或 None | Q 读取 state 时的缩放，None 表示 `Dk**-0.5` |

所有 tensor 必须同 device、contiguous。input 只读；两个 output 不得与任何 input 或彼此共享 storage。reference 返回 None，输出均通过函数参数写回。每次调用从传入的 initial_state 开始，不依赖全局状态。

```python
from kernels.kda_megakernel import make_inputs, reference

arguments = make_inputs("kda_07", device="cpu")
reference(**arguments)
out = arguments["out"]
final_state = arguments["final_state"]
```

生成器返回的字典只是显式实参的装载方式；计算函数没有 opaque 字典接口。后续 candidate 实现采用相同函数签名。

## 两个 baseline

[baseline.py](baseline.py) 提供两个与 `reference` **参数名称、顺序、默认值、类型标注和返回值一致**的函数：

| 函数 | 调用链 |
|---|---|
| `baseline_fla` | PyTorch 投影/因果卷积/SiLU/QK L2Norm → FLA `chunk_kda` → PyTorch 门控 RMSNorm/输出投影/残差 |
| `baseline_flashkda` | PyTorch 投影/因果卷积/SiLU → 原始 `flash_kda.fwd`（含 QK L2Norm）→ PyTorch 门控 RMSNorm/输出投影/残差 |

两者都返回 None，只写 `out` 和 `final_state`。参考 LACP `lacp/la_ops/kda/three_way.py` 的模块划分和 FlashKDA state 布局转换，前后处理按本 case 的参数在 PyTorch 中实现。投影使用输入 dtype 的 PyTorch GEMM；卷积在 FP32 中累加，SiLU 后转回 act，避免增加一次 SiLU 之前的低精度舍入。

```python
from kernels.kda_megakernel import make_inputs
from kernels.kda_megakernel.baseline import baseline_fla, baseline_flashkda

arguments = make_inputs("kda_20", device="cuda")
baseline_fla(**arguments)
baseline_flashkda(**arguments)
```

这段示例依次覆盖相同输出缓冲区；需要比较结果时分别保存输出或使用独立实参副本。两个 baseline 不修改输入或 initial_state。

FLA 直接接收原始已激活的 alpha_log/beta，关闭其 gate/beta 再激活与内部 QK normalization。小于 32 的 head dimension 通过零填充适配 Tensor Core tile；使用原始 Dk 的 scale，输出与 state 裁回原尺寸。这里调用 FLA 自身的 `chunk_kda` 实现，绕过其可选第三方 backend dispatch。

FlashKDA 的原生核心要求 BF16、Dk=Dv=128，且固定 `l2_eps=1e-6`。本 adapter 支持原始 Dk/Dv≤128 的 BF16 输入：Q/K 根据 Hv/Hqk 复制，head dimension 零填充到 128，最终输出和 state 裁回原尺寸。初始 state 从 `[B,Hv,Dk,Dv]` 转为 FlashKDA 的 value-major 格式，最终 state 再转回。

FlashKDA 接口接收 gate/beta logits，因此 adapter 在调用内计算 `logit(alpha_log/-5)` 和 `logit(beta)`，设置 A_log=0、dt_bias=0、lower_bound=-5。合法范围为 `alpha_log∈[-5,0]`、`beta∈[0,1]`；端点保留为 ±Inf logits，不通过 clamp 改变输入。logits 转 BF16 及 FlashKDA 内部近似 sigmoid 会引入数值误差，不能认为是精确逆变换。FP16/FP32 激活、Dk/Dv>128、其他 l2_eps 或越界 gate 会明确报错，不静默转换输入精度或替换后端。

两个 baseline 都需要 CUDA 和对应已安装的 FLA/FlashKDA 包；import 本模块不要求加载这两个可选包。任何 packing、padding、gate 转换、workspace 分配与 state 布局转换都在函数调用内部，属于端到端计时范围。

FlashKDA 扩展必须与当前 PyTorch/CUDA 环境兼容。本机原先安装的 `flash_kda_C` 链接 CUDA 12，而 `linear` 中 PyTorch 使用 CUDA 13；LACP 用下列维护脚本重新构建原始 FlashKDA 源码，不覆盖原有安装：

```bash
export LACP_ROOT="$HOME/yhy/llm/LACP"
python "$LACP_ROOT/scripts/build_glm53_flashkda.py" \
    --cache tmp/flashkda-builds --path-file tmp/flashkda-build-path.txt
export PYTHONPATH="$(cat tmp/flashkda-build-path.txt):${PYTHONPATH:-}"
```

在新的 Python 进程中运行 baseline，使其加载兼容扩展。该构建脚本针对本任务的 H20/SM90a；其他设备需要匹配其实际架构的构建。

## Reference 数学与精度

reference 将多个 PyTorch 算子串联，不是现有 CUDA megakernel 的封装。

1. FP32 计算投影，结果存为 act；零左填充因果 depthwise conv 在 FP32 计算，经过 SiLU 后存为 act。
2. Q/K 在 FP32 中按最后一维执行 `x / sqrt(sum(x*x)+l2_eps)`，再存为 act。grouped value heads 使用 `repeat_interleave(Hv/Hqk)` 映射 Q/K。
3. 下列 token recurrence 在 FP32 计算。对每个 batch/value head，令 state 为 `[Dk,Dv]`：

   ```text
   S_decay = diag(exp(alpha_log_t)) @ S_previous
   delta   = beta_t * (v_t - k_t^T @ S_decay)
   S_t     = S_decay + k_t @ delta^T
   o_t     = scale * q_t^T @ S_t
   ```

   state 不在 token 之间转成低精度；attention 输出存为 act。
4. 每个 value head 独立执行 FP32 RMSNorm、乘 norm_weight、乘 sigmoid(gate)，结果存为 act。
5. FP32 输出投影加 residual，最后存入 act 的 out；final_state 存 FP32。

这是同一 KDA 数学递推的透明 reference。LACP 的实际顺序实现使用 chunk 内三角求解、BF16 W/U、chunk-entry state operand 等额外舍入边界；本 reference **不声称**与其位级一致或已通过其历史 23 个中间变量门限。新 candidate 应按本 case 明确的语义和当前任务精度要求比较；本目录没有伪造 candidate 准确率或性能结果。

在 GPU 上执行 FP32 matmul 时，TF32/全局 matmul policy 由调用方设置并记录；reference 不静默修改全局策略。token recurrence 的 Python 循环是验证路径，大 case 运行时间可能较长，不能将它的速度当作现有 LACP 顺序 CUDA baseline 的速度。

## 20 组 inputs

完整配方见 [cases.json](cases.json)，生成器见 [inputs.py](inputs.py)。表中 head 为 `Hqk/Hv`，dim 为 `Dk/Dv`。全部使用 contiguous layout、Kc=4；alpha_log/initial_state/final_state 为 FP32，其余 tensor 为表中 act。

| ID | B × S × M | head | dim | act | X 分布 / 重点 | seed |
|---|---|---|---|---|---|---|
| kda_01 | 1 × 1 × 16 | 1/1 | 8/8 | FP32 | Gaussian，单 token/卷积左边界 | 42 |
| kda_02 | 1 × 7 × 24 | 2/2 | 8/8 | FP32 | Uniform[-1,1]，短奇数序列 | 43 |
| kda_03 | 1 × 15 × 32 | 2/2 | 16/8 | FP16 | Gaussian，16 前边界、Dk≠Dv | 44 |
| kda_04 | 1 × 16 × 32 | 2/2 | 16/16 | BF16 | Gaussian，16 边界 | 45 |
| kda_05 | 1 × 17 × 48 | 3/3 | 8/16 | FP32 | ±1 等概率，三个 heads | 46 |
| kda_06 | 1 × 63 × 32 | 2/2 | 16/16 | BF16 | Gaussian，64 前边界 | 47 |
| kda_07 | 1 × 64 × 32 | 2/2 | 16/16 | FP16 | Gaussian，64 边界 | 48 |
| kda_08 | 1 × 65 × 32 | 2/2 | 16/16 | FP32 | Gaussian，64 后边界 | 49 |
| kda_09 | 2 × 31 × 24 | 1/2 | 8/12 | FP32 | Uniform，batch/GVA/非 2 次幂 Dv | 50 |
| kda_10 | 2 × 33 × 32 | 2/4 | 16/8 | BF16 | Gaussian，grouped value heads | 51 |
| kda_11 | 1 × 127 × 32 | 2/2 | 16/16 | FP32 | Gaussian，128 前边界 | 52 |
| kda_12 | 1 × 128 × 32 | 2/2 | 16/16 | BF16 | Gaussian，弱衰减/非零状态 | 53 |
| kda_13 | 1 × 129 × 32 | 2/2 | 16/16 | FP16 | Gaussian，强衰减 | 54 |
| kda_14 | 1 × 96 × 32 | 2/2 | 16/16 | FP32 | X=0，精确 residual passthrough | 55 |
| kda_15 | 1 × 48 × 32 | 2/2 | 16/16 | FP32 | 5% 稀疏 Gaussian | 56 |
| kda_16 | 1 × 64 × 32 | 2/2 | 16/16 | BF16 | 1e-3 Gaussian，alpha=beta=0，非零状态 | 57 |
| kda_17 | 1 × 64 × 32 | 2/2 | 16/16 | FP16 | Gaussian 的 1% 元素放大 32 倍，beta=1 | 58 |
| kda_18 | 1 × 256 × 64 | 4/4 | 16/16 | BF16 | 交替衰减、beta∈{0,1}、gate scale=8 | 59 |
| kda_19 | 1 × 512 × 128 | 4/4 | 32/32 | BF16 | Gaussian，较长序列/非零状态 | 60 |
| kda_20 | 1 × 65536 × 4096 | 64/64 | 128/128 | BF16 | 原始 GLM-5.3-Flash 64K shape/分布族 | 42 |

默认配方逐输入为：

- X：表中分布与 input_scale；随机值先以 FP32 生成再转换成 act。
- in_proj_qkv_w：Gaussian，std=`1/sqrt(M)`；o_proj：Gaussian，std=`1/sqrt(Hv*Dv)`；conv1d_weight：Gaussian，std=0.2。
- alpha_log：`-Uniform[0,1)*decay_scale`，默认 scale=1；kda_12 为 0.001，kda_13 为 5；kda_18 在偶数/奇数 token 上分别乘 0.001/3。
- beta：Uniform[0,1)，特殊 case 按表设置；gate：Gaussian，默认 std=1。
- norm_weight=1；initial_state 默认为 0，非零状态 std=0.05；residual 为 std=1 的 Gaussian。
- out/final_state 预填 NaN，用于检查是否完整写回。

kda_20 的两个投影权重和 conv weight 都用 std=1，其他值族与 LACP 原始 factory 对应。它使用本地生成配方，不声称复现原 factory 的 RNG 消耗顺序或逐元素相同输入。

生成器使用独立、按 case seed 初始化的 `torch.Generator`，不改变调用者全局 RNG。相同 backend/device/PyTorch 下可重复；跨 CPU/CUDA 不保证逐元素相同，要做 backend 对照时应共享已生成的 tensor 值。

20 组都是这个通用 reference 的合法输入。旧的固定 GLM CUDA 版本可能不支持其他 shape、dtype 或 tail；适配 candidate 时需要声明其支持域，不能伪装成已有 megakernel 已支持全部输入。

## 运行

在仓库根目录、已安装 PyTorch 的环境执行：

```bash
python -m kernels.kda_megakernel.run --list
python -m kernels.kda_megakernel.run --validate-inputs
python -m kernels.kda_megakernel.run --case kda_07 --device cpu
```

完整 64K case 需要显式选择并允许大输入：

```bash
python -m kernels.kda_megakernel.run --case kda_20 --device cuda --include-large
```

`--case all --include-large` 会实际运行全部 20 组，不会静默缩小或跳过生产 shape。`--validate-inputs` 使用 meta tensors，只验证接口/shape/dtype，结果明确标为非数值运行。

## 优化版本的精度 verification

每个优化版本与本目录 reference 使用相同 Python 接口，对 `cases.json` 的全部 20 组确定输入分别运行。每组共享相同的输入值与 initial_state，使用独立输出缓冲区，比较 **out 和 final_state 的全部元素**。

按用户设置的统一或逐输出精度阈值判定：两个输出都通过，该输入为 `pass`；任一输出不满足要求，该输入为 `fail`。最终返回 20 个输入 ID 对应的 pass/fail，执行方法见 [kernel-verification](../../skills/kernel-verification/SKILL.md)。

## 来源

原项目路径相对 `LACP/`，不作为本目录运行依赖：

- `lacp/la_ops/kda/kda_attn.py::kda_attn_fwd_no_memcpy`：完整模块边界、权重布局、因果卷积、normalization、gate 与 residual 顺序。
- `lacp/la_ops/kda/kda_attn.py::prepare_kda_inputs`：输入/状态分布及原始 GLM 64K case。
- `lacp/la_ops/kda/config.py::KDA_CONFIGS['glm5-3_flash']`：4096 hidden、64 heads、128 dims、conv4、RMSNorm eps。
- `lacp/la_ops/kda/megakernel_agent/dsv41/reference.py`：等价 chunk KDA 数学与状态方向；本目录改写为 token recurrence。
- `lacp/la_ops/gated_delta_rule/fused_norm_gate.py`：RMSNorm 后乘 sigmoid gate。

读取于 2026-11-19。源码 SHA256：

```text
62278d051522856ae16987e6c653cb218c8a6d49cfb933206770302789a50102  lacp/la_ops/kda/kda_attn.py
be003011c8952cbcaf3a43eff3f64238d2860e2e9fb304afb60c6da47340c139  lacp/la_ops/kda/config.py
caa25f2d7de227c778133e45756b636cca92e9e16f077157bd616099b7b3451b  lacp/la_ops/kda/megakernel_agent/dsv41/reference.py
a0f88992e3155b2cc5dc983de3d4cd5ba964f8a2a688bd9ce801b5457423d292  lacp/la_ops/gated_delta_rule/fused_norm_gate.py
```
