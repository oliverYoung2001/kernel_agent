---
name: prepare-environment
description: Build and activate the node-specific Conda linear and Spack environment, including the current qcjz/qhjz/bjdb cluster and fuse, CUDA 13.2, and FlashKDA compiled for the GPU visible on the compute node.
---

# 构建与激活 Conda / Spack 环境

使用本目录的 [build.sh](build.sh) 一键构建环境，使用 [envs.sh](envs.sh) 激活环境。配置依据 `~/yhy/llm/LACP/scripts/envs.sh`；构建版本补充核对了 `scripts/install.sh` 和本机实际安装。

`envs.sh` 在实际执行节点调用 `hostname`，统一管理以下分支；在 Slurm 中应在 `srun` 启动的 compute-node shell 内 source。`build.sh` 仍负责下面的当前集群配置，fuse 使用已有环境。

| hostname 匹配 | Conda 默认根目录 | Spack / CUDA 激活 |
|---|---|---|
| `*fuse*` | `$HOME/.local/miniconda3` | `linear`、`gcc@12.2.0`；存在时使用 `/usr/local/cuda-13.2`；`OMP_NUM_THREADS=8` |
| `*qcjz*` / `*qhjz*` / `*bjdb*` | `$HOME/anaconda3` | 初始化 Conda/Spack，加载 `numactl@2.0.18`，退出两层 Conda，再激活 `linear`，设置 Conda CUDA 13.2 的 bin/lib/CCCL 路径并检查 nvcc，加载 `gcc@13.4.0` |

当前登录节点实际名为 `qhjz-cpu-server`，因此同时保留请求中的 `qcjz` 和实际的 `qhjz`。两处根目录均支持 `KERNEL_CONDA_ROOT` / `KERNEL_SPACK_ROOT` 覆盖；其他 hostname 保留原来的当前集群默认配置，便于显式指定根目录迁移。`kernels/kda_megakernel/tools/env.sh` 与 `kernels/kda_module_dist/tools/env.sh` 都调用此入口并设置仓库 `PYTHONPATH`。

分布式 KDA 还需要支持 `fla.ops.cp.build_cp_context`、`chunk_kda(cp_context=...)` 及 CP causal convolution 的 FLA 版本。仅 `import fla` 成功不能证明 CP 接口可用；在分配 GPU 的节点检查：

```bash
source skills/prepare-environment/envs.sh
python -c 'import inspect; from fla.ops.cp import build_cp_context; from fla.ops.kda import chunk_kda; from fla.modules.convolution import causal_conv1d; print(inspect.signature(build_cp_context)); print(inspect.signature(chunk_kda)); print(inspect.signature(causal_conv1d))'
```

## 环境配置

| 组件 | 安装/提供方式 | 版本或位置 |
|---|---|---|
| Conda 环境 | Conda | `linear`，Python 3.12.12 |
| PyTorch | 环境内 pip | `torch==2.11.0`，实际 build 为 `2.11.0+cu130`，runtime 13.0 |
| torchvision / torchaudio | 环境内 pip | `0.26.0` / `2.11.0` |
| CUDA nvcc / crt / runtime / cccl / nvvm | 环境内 pip | `13.2.86` |
| CUDA_HOME | Conda 内的 CUDA 包目录 | `$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13` |
| causal_conv1d | 激活 `linear` 后用 pip 安装 | 本机实测版本 `1.7.0` |
| FlashKDA | 环境内从源码编译安装 | `MoonshotAI/FlashKDA`，从计算节点当前可见 GPU 探测目标架构 |
| numactl | Spack | `numactl@2.0.18` |
| GCC | Spack | `gcc@13.4.0` |

原脚本中的 `spack load cuda@12.8.1` 已被注释，不启用它。CUDA 工具链使用上述 Conda 内的 `nvidia/cu13`。PyTorch runtime 13.0 与 nvcc 13.2 是来源环境的实际组合。

## 首次构建

已有 `linear` 环境及对应 Spack 包时，直接执行下一节的激活命令。

本机 Conda 位于 `$HOME/anaconda3`，Spack 位于 `$HOME/.local/spack`。以下命令在已安装 Conda 和 Spack 的机器执行；迁移时可用 `KERNEL_CONDA_ROOT` / `KERNEL_SPACK_ROOT` 指定其安装根目录。

可直接使用脚本的构建入口；检测到已有 `linear` 时会退出，不覆盖：

```bash
bash skills/prepare-environment/build.sh
```

脚本执行的构建命令为：

```bash
source "${KERNEL_CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh"
source "${KERNEL_SPACK_ROOT:-$HOME/.local/spack}/share/spack/setup-env.sh"

spack install numactl@2.0.18
spack install gcc@13.4.0

conda create -n linear python=3.12.12 pip -y
conda activate linear

python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
python -m pip install \
    nvidia-cuda-nvcc==13.2.86 \
    nvidia-cuda-crt==13.2.86 \
    nvidia-cuda-runtime==13.2.86 \
    nvidia-cuda-cccl==13.2.86 \
    nvidia-nvvm==13.2.86
```

先安装 PyTorch，再安装来源环境对应的 CUDA 13.2 组件。脚本随后加载同目录的 `envs.sh`，使 CUDA 和 GCC 就绪，再执行下一节的 FlashKDA 安装。最后检查 Python、nvcc、GCC、numactl、PyTorch 和 FlashKDA 导入，并打印完成信息。脚本通过自身位置定位文件，可从任意工作目录运行。

## 安装 causal_conv1d

在已有的 `linear` 环境中，或在首次运行 `build.sh` 完成后，执行以下步骤安装因果卷积扩展并检查前向、反向接口。已有安装时可直接执行导入检查。

```bash
conda activate linear
python -m pip install causal_conv1d
python -c 'from causal_conv1d.cpp_functions import _causal_conv1d_fwd_cpp, _causal_conv1d_bwd_cpp'
```

## 安装 FlashKDA

`build.sh` 会将 FlashKDA 安装到新建的 `linear` 环境中。源码默认放在 `$HOME/flash-kda`，可通过 `KERNEL_FLASH_KDA_ROOT` 指定其他路径；目录已存在时复用源码，不重新克隆。

已有 `linear` 环境时，在本仓库根目录执行以下命令单独补装 FlashKDA（已有源码则跳过 `git clone`）：

```bash
source skills/prepare-environment/envs.sh
python -m pip install setuptools wheel ninja
git clone https://github.com/MoonshotAI/FlashKDA.git "${KERNEL_FLASH_KDA_ROOT:-$HOME/flash-kda}"
cd "${KERNEL_FLASH_KDA_ROOT:-$HOME/flash-kda}"
git submodule update --init --recursive
python -m pip install -v --no-build-isolation .
python -c 'import flash_kda; print("FlashKDA:", flash_kda.__file__)'
```

在分配到 GPU 的计算节点执行安装。FlashKDA 默认读取当前 CUDA 设备的计算能力；H100 PCIe 和 H20 均生成 `sm_90a`。需要交叉编译时可显式设置 `FLASH_KDA_CUDA_ARCHS`。`--no-build-isolation` 使用已激活的 `linear` 环境中的 PyTorch 和构建依赖；编译仍需要该环境的 CUDA Toolkit 和 GCC。

## 每次激活

在本仓库根目录执行：

```bash
source skills/prepare-environment/envs.sh
```

脚本依次初始化 Conda/Spack、加载 numactl、退出两层 Conda 环境、激活 `linear`、设置 CUDA 路径、加载 GCC。原脚本依赖父 shell 已初始化 Conda/Spack，本脚本补上这两个初始化入口，供独立 shell 和作业脚本使用。

激活时设置：

```bash
export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include/cccl:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}"
```

在其他机器使用不同安装位置：

```bash
export KERNEL_CONDA_ROOT=/path/to/anaconda3
export KERNEL_SPACK_ROOT=/path/to/spack
source skills/prepare-environment/envs.sh
```

在 Bash 作业脚本中使用同一条 `source` 命令，然后执行该项目的程序。`bash build.sh` 负责首次构建，`source envs.sh` 负责激活父 shell。激活脚本支持调用方的 `set -u`：初始化和激活期间暂时关闭 nounset，结束后恢复。

## 检查激活结果

```bash
printf '%s\n' "$CONDA_DEFAULT_ENV" "$CONDA_PREFIX" "$CUDA_HOME"
python --version
nvcc --version
gcc --version
numactl --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
python -c 'import flash_kda; print(flash_kda.__file__)'
```

预期分别为 `linear`、Python 3.12.12、nvcc 13.2、GCC 13.4.0、numactl 2.0.18、PyTorch `2.11.0+cu130 / 13.0`，以及 `linear` 环境内的 FlashKDA 模块路径。这里只检查环境构建/激活与扩展导入；不运行模型、kernel 测试或 benchmark。
