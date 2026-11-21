# KDA 实验复现

在分配到的计算节点上，从仓库根目录执行 `source kernels/kda_megakernel/tools/env.sh`。脚本按当前 hostname 选择已有环境：

- 包含 `fuse` 时激活 `linear`，加载 Spack GCC 12.2.0，并在已安装时使用系统 CUDA 13.2。
- 包含 `qcjz` 或 `bjdb` 时复用仓库的 [LACP 环境脚本](../../../skills/prepare-environment/envs.sh)，加载 `linear`、numactl 2.0.18、GCC 13.4.0 和 Conda 内的 CUDA。
- 在其他节点上沿用调用者已激活的环境。所有分支都将当前仓库加入 `PYTHONPATH`。

默认 `cases/cases.json` 运行完整 attention，支持 reference、FLA 和 FlashKDA。FlashKDA 在分配的计算节点按当前可见 GPU 的计算能力构建。

各入口接收 `inpx`、`out`、`final_state` 与模型权重。Dense batch 直接传入。Packed case 使用共享 offsets 的 jagged `inpx` 与 `out`，runner 根据清单构建这两个视图。融合卷积按序列 ID 隔离历史，FLA/FlashKDA 各执行一次完整后端。CUDA reference 用 FP32 逐 token 递推，状态在独立样本边界重置。

```bash
source kernels/kda_megakernel/tools/env.sh
export KDA_EXPERIMENT=experiment1
python -m kernels.kda_megakernel.run --validate-inputs
python -m kernels.kda_megakernel.run --mode verify
```

新一轮独立实现开始时运行以下命令，创建下一个编号目录并选中该实验。续跑时设置 `KDA_EXPERIMENT` 为已有目录名。未设置时 runner 选择编号最大的实验。

```bash
KDA_EXPERIMENT=$(python -m kernels.kda_megakernel.run --new-experiment)
export KDA_EXPERIMENT
```

## 移出的旧函数

- `reference.py`：移除 `_run_module`、`run_packed`、`legacy_reference` 和 `validate_legacy_arguments`；合并 `_activated_gates`、`_normalize_qk`、`_pytorch_conv_silu` 和 `_recurrent_kda` 的职责。
- `baseline.py`：移除 `legacy_baseline_fla`、`legacy_baseline_flashkda`、`_checked_logit`、`_finish`、`_flash_logits`、`_native_conv_silu`、`_native_flash_recurrence`、`_normalize`、`_pad_last`、`_pad_state`、`_prepare_qkv` 和 `_use_fused_path`。`_normalize_packed_view` 移入 `reference.py` 供共享 L2Norm 使用。
- `cases/inputs.py` 和 `run.py`：将 `_make_module_inputs` 并入 `make_inputs_from_recipe`，移除 `verify_frozen_sources`。
- 旧 `optimized.py` 的 `optimized()` 和 `experiments/compare_baselines.py` 的 `main()` 随旧文件归档。

## 历史复现

`cases_0.json`、v0–v5 和对应数据属于[历史报告](../experiment0/reports/report.md)。[历史源码归档](../experiment0/reports/history/before-single-call.zip) 内的 runner 可执行旧接口。重跑历史入口时，把 `experiment0/v0` 到 `v5` 放入独立工作副本的 `kernels/kda_megakernel/`，并在工作副本根目录解压归档。
