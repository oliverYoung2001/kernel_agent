---
name: prepare-environment
description: Build and activate the Conda linear environment and Spack numactl 2.0.18 and GCC 13.4.0 environment used by LACP scripts/envs.sh, with the CUDA 13.2 compiler installed inside Conda.
---

# 构建与激活 Conda / Spack 环境

使用本目录的 [build.sh](build.sh) 一键构建环境，使用 [envs.sh](envs.sh) 激活环境。配置依据 `~/yhy/llm/LACP/scripts/envs.sh`；构建版本补充核对了 `scripts/install.sh` 和本机实际安装。

## 环境配置

| 组件 | 安装/提供方式 | 版本或位置 |
|---|---|---|
| Conda 环境 | Conda | `linear`，Python 3.12.12 |
| PyTorch | 环境内 pip | `torch==2.11.0`，实际 build 为 `2.11.0+cu130`，runtime 13.0 |
| torchvision / torchaudio | 环境内 pip | `0.26.0` / `2.11.0` |
| CUDA nvcc / crt / runtime / cccl / nvvm | 环境内 pip | `13.2.86` |
| CUDA_HOME | Conda 内的 CUDA 包目录 | `$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13` |
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

先安装 PyTorch，再安装来源环境对应的 CUDA 13.2 组件。构建后脚本自动加载同目录的 `envs.sh`，检查 Python、nvcc、GCC、numactl 和 PyTorch，并打印完成信息。脚本通过自身位置定位文件，可从任意工作目录运行。项目自己的 Python 包和 CUDA 扩展按项目要求安装。

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
```

预期分别为 `linear`、Python 3.12.12、nvcc 13.2、GCC 13.4.0、numactl 2.0.18 和 PyTorch `2.11.0+cu130 / 13.0`。这里只检查环境构建/激活；不运行模型、kernel 测试或 benchmark。
