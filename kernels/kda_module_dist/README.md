# kda_module_dist

这是 [kda_megakernel](../kda_megakernel/README.md) 完整 workload 的分布式版本，计算范围仍为输入 RMSNorm → 合并投影 → QKV 因果卷积及 SiLU → KDA → gate 投影及门控 RMSNorm → 输出投影 → 残差。三个 baseline 都按展平的 `batch * sequence` 顺序连续、均匀切分输入与输出，只有 `out` 是公开输出，无 `final_state` 参数。函数的并行度来自显式 `group`，不硬编码 8 或 16；20 个生产配方使用 8/16 卡。

## 显式接口

[reference.py](reference.py) 的 `reference` 和 [baseline.py](baseline.py) 的 `baseline_tp`、`baseline_ulysses`、`baseline_kcp` 使用相同签名：

```python
def operation(
    inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
    f_b_proj_weight, g_b_proj_weight, a_log_weight, dt_bias_weight, norm_weight,
    o_proj_weight, cu_seqlens, hq, hv, dk, dv, group=None,
): ...
```

`inpx`、权重、`cu_seqlens` 为 input；`out` 为 caller-owned output。初始 recurrence state 恒为零，序列之间重置；函数返回 `None`。内部需要的 recurrence scratch 不作为输出。`group=None` 使用 WORLD，未初始化 distributed 时仅支持单进程调用。所有 rank 必须以相同顺序调用同一实现，使用相同 global boundaries、权重配方和 scalar 参数；子 group 的 rank 顺序决定 token/head 顺序。

符号：`P` 为 group size，`r` 为 group 内 rank，`T` 为总 token 数，`L=T/P`，`M` 为 hidden，`Hq=hq`，`Hv=hv`，`K=dk`，`V=dv`，`W` 为 convolution width，`C=2*Hq*K+Hv*V`。要求 `T%P=0`、`Hv%Hq=0`；TP/Ulysses 还要求 `Hq%P=Hv%P=0`。KCP 不要求 head 数能被 P 整除，但当前 native CP 要求 `K<=256`；多 rank CP convolution 要求 `L>=W-1`。生产配方使用 BF16、`K=V=128`、`W=4`。native FLA baseline 的激活支持域是 BF16/FP16，oracle 还允许 FP32。

| 参数 | reference / Ulysses / KCP shape | TP shape | dtype |
|---|---|---|---|
| `inpx`, `out` | `[L,M]` | 同左 | activation |
| `input_norm_weight` | `[M]` | 同左，复制 | activation |
| `in_proj_qkvbfg_a_weight` | `[M,C+Hv+K+V]` | `[M,C/P+Hv/P+K+V]` | activation |
| `conv1d_weight` | `[C,1,W]` | `[C/P,1,W]` | activation |
| `f_b_proj_weight` | `[K,Hv*K]` | `[K,Hv*K/P]` | activation |
| `g_b_proj_weight` | `[V,Hv*V]` | `[V,Hv*V/P]` | activation |
| `a_log_weight` | `[Hv]` | `[Hv/P]` | FP32 |
| `dt_bias_weight` | `[Hv,K]` | `[Hv/P,K]` | FP32 |
| `norm_weight` | `[V]` | 同左，复制 | activation |
| `o_proj_weight` | `[Hv*V,M]` | `[Hv*V/P,M]` | activation |
| `cu_seqlens` | `[N+1]`，global 累积边界 | 同左，复制 | int64 |

所有 tensor 为 contiguous、同 device；`out` 不得 alias 任何输入。`cu_seqlens[0]=0`、末项为 T，严格递增。定长 batch 也显式传入 `[0,S,2S,...,B*S]`；变长传入实际累积边界。rank r 负责 `[r*L,(r+1)*L)`，因此一个序列可以跨多个 rank，一个 rank 也可能含多个序列。

TP 不是将整个 merged weight 一刀切成 P 份：分别切 Q、K、V、beta 的 head 区域，再保留复制的低秩 `f_a[K]` 与 `g_a[V]` 因子。输出投影按行切分。具体索引见 [distributed.py](distributed.py) 的 `weight_indices`；`partition_weights` 可从完整权重得到相同 shard。TP 执行入口只接收上述 shard，不接收完整 head 权重。

## Reference 与三个 baseline

`reference` all-gather 全局 `inpx`，使用原模块的独立 FP32 token recurrence oracle（以 BF16/FP16 保存投影、卷积、归一化结果），然后写回本 rank 的 output slice。oracle 仅用于定义结果，成本高于 baseline；其内部 scratch state 不对外暴露。

| 实现 | 实际执行顺序 | 权重 |
|---|---|---|
| `tp` | token all-gather → 本 rank head 的完整 `_project`、KDA 与 gate → `torch.mm(..., out=partial)` → token reduce-scatter SUM → local residual | head shard + 必需的复制向量/低秩因子 |
| `ulysses` | local `_project`（含 convolution halo）→ Q/K/V/decay/beta 各做一次 all-to-all，从 token 分片转为 head 分片 → 完整序列 KDA → attention all-to-all 回 token 分片 → local gate/output/residual | 复制；recurrence 的 A_log/dt_bias 取本 rank head slice |
| `kcp` | local `_project`（含 convolution halo）→ native FLA CP KDA → local gate/output/residual | 复制 |

Ulysses 的 `g_a` 是 `[1,L,V]` 低秩 output gate 因子，没有 head 维；保留在本 rank，在 attention 转回 token 分片后才扩展，不参与 core KDA。Q/K 与 V 的 head 分组分别按 Hq/Hv 切分，保留 GVA 对应关系。

SP/KCP 调用 `fla.modules.convolution.causal_conv1d(cp_context=...)` 传递 `W-1` 个历史 token；不能直接对每个 token shard 独立做零初始卷积。FLA 的 context 同时处理 rank 间 continuation 和 shard 内的真实 sequence reset。KCP 使用 `build_cp_context(global_cu_seqlens, group, conv1d_kernel_size=W)` 和 `chunk_kda(cp_context=..., output_final_state=False)`；该 CP 路径 all-gather 每个 rank 的 FP32 affine state/transition `[Hv,K,V+K]`，然后按真实序列的前驱链合并 incoming state，无全量 QKV all-gather。机制来源为 [FLA CP README](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md)。

当前 KCP 对齐的 FLA 源码版本为 `0d42d7ee9ffe88f42f7e2b08d81cd0f5b067f052`，目标调用为该 checkout 的 `tests/cp/test_kda.py::_run_chunk`。环境准备见 [prepare-environment](../../skills/prepare-environment/SKILL.md)。TP/SP 的普通 KDA 同样用 FLA；不依赖 FlashKDA。

### 与 FLA CP 完全一致的核心接口

[kcp.py](kcp.py) 直接导出 FLA 的公开 `chunk_kda`、`build_cp_context` 函数对象，保留全部参数、默认值、dispatch、校验、返回 tuple 和 autograd。这里没有复制一份 CP 内核；两入口使用同一组计算与通信算子。核心入口接受 Q/K/V/g/beta；前面的 `baseline_kcp(inpx, out, ...)` 是完整 module 适配层。

```python
from kernels.kda_module_dist.kcp import build_cp_context, chunk_kda

context = build_cp_context(cu_seqlens=global_cu_seqlens, group=group,
                           layout="contiguous")  # also supports "zigzag"
o, final_state = chunk_kda(
    q=q_local, k=k_local, v=v_local, g=raw_gate_local,
    beta=activated_beta_local,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    disable_recompute=False, state_v_first=False,
    cp_context=context, A_log=A_log, dt_bias=dt_bias.flatten(),
)
# final_state is None in CP mode; o and input gradients follow the FLA API.
```

核心接口也支持目标测试中的 pre-gated `g`、`disable_recompute=True`、`state_v_first=True` 和 legacy `use_tf32x3_affine_chain=True` context 参数。CP 的 `initial_state` 必须为 `None`、`output_final_state=False`；限制与上游完全相同。普通非 CP 调用保留原生功能。核心接口本身不隐式改变 beta；默认 beta 已激活。完整 module 适配层接收投影得到的 beta logits，使用 activation dtype 的 `beta.sigmoid()`，并将原始 Q/K/g 直接交给内核内 L2 norm 和融合 gate+cumsum。此前 KCP 的外部 norm、独立 gate 和 FP32 beta sigmoid 路径已移除；TP/Ulysses 保持现有调用方式。

分区和并行算法由同一个 FLA context 和 CP 算子执行：contiguous 每 rank 取 `T/P` 个连续 token；zigzag 将全局序列切成 `2P` 段，rank r 拼接第 r 段和第 `2P-1-r` 段。真实 sequence 边界及 rank/zigzag 分段边界形成 local fragments；默认每个 fragment 从自己的起点按 64-token chunk 切分，不补齐成统一的全局 chunk 网格。完整 module 输入仍使用原先的 contiguous 分片；zigzag 通过核心接口使用。

每个 chunk 的仿射 recurrence 是 `S_next = (D - KGᵀW) S + KGᵀU`。CP 先计算跨 rank continuation 所需 fragment 的零初始响应和 transition，all-gather FP32 `[Hv,K,V+K]` summaries，再沿真实序列的前驱顺序合并 incoming state，最后用 incoming state 执行本地 chunk recurrence；反向沿后继链处理。zigzag 交换两份 summary，并按 fronts 正序、backs 倒序恢复全局依赖。CUDA 的 affine transition 连乘/merge 使用 FP32 tensor 和 `tf32x3` dot；没有新增 P2P pipeline 或自定义 NCCL 算子。卷积仍使用 FLA CP causal convolution 的 halo all-gather。

计算入口保留 FLA 的 backend dispatch：L2 norm、gate/cumsum、chunk intra/WY、CP preprocess/merge、state recurrence 和 output 使用其现有算子；WY fused backward 还保留已注册的 TileLang backend（目标等 head 测试满足其 verifier，GVA 则回退 Triton）。CP 通信经 `fla.ops.cp.comm.all_gather_into_tensor` 调用 `torch.distributed.all_gather_into_tensor`，本次 CUDA 测试的 process group 使用 NCCL。KCP 不再通过 `inspect.unwrap` 绕过公开入口。

目标测试的 BF16 路径数值类型如下；FP32 accumulation 不代表全部 scratch 都是 FP32：

| 张量/操作 | dtype / 实现 |
|---|---|
| Q/K/V、raw g、已 sigmoid 的 beta、输出及对应输入梯度 | BF16；Q/K L2 norm 内部 FP32 归约后写回 BF16 |
| pre-gated g（目标测试的另一分支）及其梯度 | FP32 |
| A_log、dt_bias、gate activation、chunk gate cumsum | FP32；gate+cumsum 融合 Triton kernel |
| WY 的 W/U、QG/KG、chunk state scratch、v_new | activation dtype；dot 内部 FP32 accumulator |
| intra 的对角 Akk scratch | FP32（其他 Aqk/Akk 为 activation dtype） |
| CP incoming state、affine summary/transition、all-gather payload | FP32；CUDA chain/merge dot 为 tf32x3 |

精度对照统一从 `run.py --mode verify` 进入，专项实现为 [verification/cp_alignment.py](verification/cp_alignment.py)。它读取目标文件的全部 18 组原始参数，复用原始随机输入、contiguous/zigzag shard、逐序列 naive recurrence reference 和阈值策略；逐 rank 比较两入口的 `o/dq/dk/dv/dg/dbeta`，并在适用 case 检查 module adapter 的非连续 QKV 视图及 beta logits 转换。两入口须逐元素相同、全部 finite；对独立 reference 的 output/dq/dk/dv 沿用上游 `8e-3` ratio 门限，dg/dbeta 沿用 warning-only，报告中单独列出实际误差。它不是两份独立 CP 内核的交叉验证。

```bash
source kernels/kda_module_dist/tools/env.sh
python -m kernels.kda_module_dist.run --mode verify --cp-alignment --list
# Run on an allocated node with at least eight GPUs; the suite spawns 2/4/8 ranks.
python -m kernels.kda_module_dist.run --mode verify --cp-alignment \
  --experiment kernels/kda_module_dist/experiment1 --resume
# The maintained Slurm launcher also runs the complete-module regression checks.
sbatch --wait --output=kernels/kda_module_dist/experiment1/verification/slurm-%j.log \
  kernels/kda_module_dist/verification/run_cp_alignment.sh
```

新精度数据保存在 `experiment1/verification/kcp_alignment/`，完整结果见 [KCP 对齐报告](experiment1/reports/kcp_alignment.md)。65,536-token 的独立 reference 通过 `torch.autograd.graph.save_on_cpu(pin_memory=True)` 卸载保存张量，避免单卡保存整张反向图时顶满 H20 显存；reference 的输入、GPU 算子、dtype 和反向图保持原样，不缩短序列。首次全 GPU 保存图的尝试及其日志单独保留。`identity.json` 固定 FLA revision、状态、目标 test 及本地实现 SHA256；续跑要求源码身份完全一致。`experiment0` 的性能报告对应旧 FLA `bb324a329e58c06367e043ce4f8b195f490f9144` 和旧 KCP 适配层，不能视为本次对齐后的性能数据。

## 20 个输入配方

[cases.json](cases/cases.json) 保存全部 dtype、layout、seed、模型 revision、数值分布和尺寸；[inputs.py](cases/inputs.py) 提供 `load_cases/get_case/make_inputs/make_inputs_from_recipe`。每个模型恰好 5 个：8 卡变长、8 卡 128K、16 卡变长、16 卡 128K、16 卡 256K。`K=1024` tokens。

| 模型 | M | Hq=Hv | case ID 后缀（按上述顺序） |
|---|---:|---:|---|
| Kimi K3.1 Pro | 7168 | 96 | `kimi_k31_pro_01`–`05` |
| GLM-5.3-Flash | 4096 | 64 | `glm53_flash_06`–`10` |
| Kimi Linear 48B-A3B | 2304 | 32 | `kimi_linear_48b_a3b_11`–`15` |
| Ling-3.1-flash | 4096 | 64 | `ling31_flash_16`–`20` |

完整 ID 加前缀 `kda_dist_`。变长配方均为 `batch=5, variable_length=true, sequence=131072, sequence_lengths=[50671,7451,9923,8622,54405]`，总和为 131072。定长配方为 `batch=1`，sequence 为 131072 或 262144。

生成配方保存在源码，不提交大型 tensor。activation 和权重用 CPU `torch.randn` 的独立 generator 生成，每 128 行使用 `case_seed + 1000003*stream + block_index`，生成后拷到指定 device；在相同 PyTorch 版本可重放，不修改 caller RNG。`inpx~N(0,1)`，投影权重标准差分别为 `M^-0.5`、`K^-0.5`、`V^-0.5`、`(Hv*V)^-0.5`；conv 标准差 0.2；norm=1，A_log=dt_bias=0，out 预填 NaN。TP 直接生成 shard：仅暂存当前随机行块并选择相关列，不物化完整权重。不同 rank 的输入拼接及 TP 权重拼接与复制版本逐元素一致。`make_inputs` 默认从 `group` 取得当前 rank，也可显式传 `rank` 做离线/meta 检查；实际分布式生成时检查 group size 与配方一致。

```bash
source kernels/kda_module_dist/tools/env.sh
python -m kernels.kda_module_dist.cases.generate_cases --check
python -m kernels.kda_module_dist.run --list
python -m kernels.kda_module_dist.run --validate-inputs --impl reference tp ulysses kcp
```

metadata 检查逐 rank 绑定真实函数，构造 meta tensor 并检查规格；不会分配大 GPU tensor。生产 case 只按需生成，必须传 `--include-large`，且启动进程数必须等于配方 `parallel_size`。

在 Slurm 中，每节点只启动一个 launcher：

```bash
# 8 GPUs: GLM, batch=1, 128K, all three baselines.
srun --unbuffered -p debug -N 1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=16 \
  bash kernels/kda_module_dist/tools/launch.sh --include-large --case kda_dist_glm53_flash_07

# 16 GPUs: GLM, batch=1, 256K.
srun --unbuffered -p debug -N 2 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=16 \
  bash kernels/kda_module_dist/tools/launch.sh --include-large --case kda_dist_glm53_flash_10
```

已有 allocation 或非 Slurm 环境，可先在 GPU 节点 source 环境，再直接 `torchrun --standalone --nproc_per_node=8 -m kernels.kda_module_dist.run ...`。分布式多节点 launch 使用每个节点的 `SLURM_NODEID`，默认 master 为 allocation 第一台节点，端口为 `20000+job_id%20000`，支持 `MASTER_ADDR` / `MASTER_PORT` 和 `KERNEL_GPUS_PER_NODE` 覆盖。

`--mode verify` 默认依照 verification skill 使用第一个 baseline TP 作 ground truth；`--ground-truth reference` 可显式选择独立 token oracle。比较全部 `out` 元素，默认要求 global scaled_max 和 rmse_ratio 均 ≤0.01，检查 finite/shape/dtype；FP64 统计先局部再跨 rank 归约。TP 的 partial output/collective 舍入、FLA chunk 求解与 token oracle 不保证 bitwise 相同。`--mode timing --warmup 5 --repeat 20 --output tmp/dist-timing.json` 保存每次完整调用的 wall-clock ms，以各 rank 的最大值为该次样本，含函数内转换/分配/collective 和输出完成，不含输入生成和验证。该 warmup/repeat 只是启动值，正式性能结论需按稳定性调整。候选实现支持 `--impl package.module:function --weight-layout tp`，使用同一显式接口。

### 三个 baseline 的正式采集与报告

20 个生产输入 × 3 个 baseline 共 60 项已完成数值验证、warmup=10/repeat=30 的普通完整调用计时及全部 rank 的独立 CUPTI 捕获；数值检查均通过。当前报告位于 [experiment0/reports/report.md](experiment0/reports/report.md)，仅包含 TP/Ulysses/KCP，不创建虚构版本。该实验的采集与生成入口保存在 [reports/](experiment0/reports/)，原始数据保存于 `experiment0/baselines/<name>/`。

`run.py --mode suite --experiment <path> --warmup 10 --repeat 30` 对指定 case 使用 TP ground truth 完成验证，独立测量未开启 profiler 的完整调用，然后另行捕获每个 rank 的 CUPTI activities。`--mode profile` 可单独采集 CUPTI；`--mode capture` 为 NCU 提供一次完整调用的 `cudaProfilerStart/Stop` 边界。共享采集实现见 [profiling/](profiling/)。`--resume` 仅跳过源码身份一致且采集完整的记录；已有完整记录若源码/采集器身份变化则拒绝覆盖，改用新的 `--experiment` 目录。生产输入始终使用原 manifest shape；小规模采集自检放在实验 `tmp/`，不混入正式报告。

正式采集的输入准备使用 `profiling/inputs.py` 在节点 `/tmp/kda-dist-report-<uid>-20261122/` 缓存不可变 CPU tensor（可用 `KDA_REPORT_INPUT_CACHE` 指定目录）。缓存键包含完整配方、PyTorch 版本及原生成器/shape helper 的 SHA256。CPU tensor 由当前配方生成，逐元素对照原生成器；每次调用准备独立的 GPU 输入、权重和输出，TP 在上传 GPU 前只保留 head shard。缓存不包含实现代码、输出结果或性能数据，不改变 `run.py --mode timing` 的普通生成器，也不进入任何正式计时区间。

## 静态结构、工作量与依赖

以下为 static-derived，不是实测时间或硬件 busy interval。此 workload 是多 kernel 的 Python module，GEMM/FLA/Triton 的 grid、block、寄存器和驻留由依赖 dispatch 决定；未采集编译资源/warp 时间线，不虚构统一 megakernel launch 或 warp 分工。主要源码锚点为 `baseline._execute/_recurrence/_finish`、KCP 的 `kcp.module_recurrence`、`distributed.sequence_to_head/head_to_sequence` 及原模块 `reference._project/convolve_qkv`。

| 阶段 | per-rank 逻辑工作 | 必要依赖 |
|---|---|---|
| input norm / projection | GEMM `2*J*M*(Cj+Hj+K+V)`；decay GEMM `2*J*K*Hj*K` | token input ready → norm → merged QKV/beta/f_a/g_a → decay/conv |
| convolution | `J*Cj*W` 次有效 tap 上限；边界 tap masked；SiLU 含 sigmoid/SFU | halo 已到达且 sequence 边界可用 → QKV ready |
| KDA | token 语义 state `[Hj,K,V]`，衰减、delta update、query readout；FP32 内部 state | TP/Ulysses 完整序列或 KCP incoming state ready → attention ready；序列间 reset |
| output gate / norm | gate GEMM `2*J*V*Hj*V`；每 head 沿 V 做 RMSNorm/sigmoid | attention 与 g_a 均 ready → gated |
| output projection | GEMM `2*J*Hj*V*M` | gated ready → partial/full output → TP reduce-scatter 完成 → local residual |

TP 为 `J=T, Cj=C/P, Hj=Hv/P`（Q/K 使用 Hq/P）；SP/KCP 投影及输出为 `J=L, Cj=C, Hj=Hv`；Ulysses core KDA 为 `J=T,Hj=Hv/P`；KCP core 为 `J=L,Hj=Hv`。矩阵乘加按 2 FLOPs 计；标量 SFU/除法不混入 Tensor FLOPs。表中矩阵为逻辑规模，padding、chunk WY 变换等执行工作取决于 FLA 源码，未声称等同实际指令量。

设 activation 字节数为 b，Ulysses 前置 payload 每个 local token 为 `b*(2*Hq*K+Hv*V+Hv*K+Hv)`，后置为 `b*Hv*V`；all-to-all 每 rank 的逻辑跨 rank send/receive 为 local payload 的 `(P-1)/P`。g_a 无通信。TP all-gather 每 rank 接收 `(P-1)*L*M*b`；reduce-scatter 输入 `[T,M]`，输出 `[L,M]`，具体链路字节取决于 NCCL 算法。KCP state all-gather 每 rank 贡献 `4*Hv*K*(V+K)` 字节，接收其余 `P-1` 份；卷积另需 `W-1` 行 QKV halo，其 native FLA 实现的物理通信以依赖版本为准。以上是逻辑 payload，不是 DRAM/shared transaction 或链路实测带宽。

## 验证范围

- 20 个生产配方可重生；全部 rank 的四种实现共 1024 组参数签名、shape/dtype/layout 已通过 meta 检查。
- CPU 3-rank Gloo 已通过：token/head all-to-all 的独立索引对照、往返、beta 无 feature 维路径、token all-gather、reduce-scatter SUM、TP 生成权重与完整权重精确切片、caller RNG 保持。
- 两张 H20 上 12 项 GPU 数值检查已通过：三个 baseline 各覆盖小 shape 的 dense、packed、弱衰减跨 rank state、输出投影为零时的残差，使用 `Hq=4,Hv=8,K=V=128,M=64,T=256`（含 GVA），并检查所有输入/权重保持不变。global scaled_max 最大 `0.004505`，rmse_ratio 最大 `0.002826`，阈值均为 0.01；零投影残差逐元素相等。小 shape 结果不代表 20 个生产 shape 已数值验证。生产输入的实际验证、计时与 profile 覆盖范围见 [报告汇总](experiment0/reports/summary.md) 和 [完整报告](experiment0/reports/report.md)。
- 同一 Slurm launcher / `run.py` 在两张 H20 上完成独立的小 shape 集成检查：packed 与 dense 各验证三个 baseline，6 项均通过并写出 JSON；另对三个 baseline 各执行 warmup=1、repeat=2 的计时输出检查。该计时仅检查运行与数据保存，不用于性能结论。

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=3 \
  -m kernels.kda_module_dist.verification.check

# Run inside an allocated two-GPU node after sourcing tools/env.sh.
torchrun --standalone --nproc_per_node=2 \
  -m kernels.kda_module_dist.verification.check --gpu
```

完成标记为 `KDA_MODULE_DIST_CHECKS_COMPLETE`。agent 自检日志放在仓库 `tmp/`，不作为 kernel 配方或正式性能数据提交。
