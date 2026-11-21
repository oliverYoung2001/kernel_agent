#!/usr/bin/env bash
# Build the Conda and Spack environment, including FlashKDA for the allocated GPU.
set -eo pipefail
set +u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_root="${KERNEL_CONDA_ROOT:-$HOME/anaconda3}"
spack_root="${KERNEL_SPACK_ROOT:-$HOME/.local/spack}"
flash_kda_root="${KERNEL_FLASH_KDA_ROOT:-$HOME/flash-kda}"

for setup_file in \
    "$conda_root/etc/profile.d/conda.sh" \
    "$spack_root/share/spack/setup-env.sh"; do
    if [[ ! -f "$setup_file" ]]; then
        printf 'Environment setup file not found: %s\n' "$setup_file" >&2
        exit 1
    fi
done

source "$conda_root/etc/profile.d/conda.sh"
source "$spack_root/share/spack/setup-env.sh"

# Check before installing anything; never replace an existing environment.
conda_envs_json="$(conda env list --json)"
existing_linear_prefix="$("$conda_root/bin/python" -c \
    'import json, os, sys; print(next((p for p in json.load(sys.stdin)["envs"] if os.path.basename(p) == "linear"), ""))' \
    <<< "$conda_envs_json")"
if [[ -n "$existing_linear_prefix" ]]; then
    printf 'Conda environment linear already exists: %s\n' "$existing_linear_prefix" >&2
    printf 'Activate with: source %q/envs.sh\n' "$script_dir" >&2
    exit 1
fi

export FLASH_KDA_CUDA_ARCHS="${FLASH_KDA_CUDA_ARCHS:-auto}"
if [[ ${FLASH_KDA_CUDA_ARCHS,,} == auto ]]; then
    if [[ -z ${CUDA_VISIBLE_DEVICES-unset} || ${CUDA_VISIBLE_DEVICES:-} == -1 ]] ||
       ! node_gpus=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader) ||
       [[ -z $node_gpus || $node_gpus == "No devices were found" ]]; then
        echo 'Run the build on the allocated GPU node or set FLASH_KDA_CUDA_ARCHS for a cross-build.' >&2
        exit 1
    fi
    printf 'Compute-node GPU inventory (model, compute capability):\n%s\n' "$node_gpus"
fi

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

# Load the toolkit before querying the CUDA device visible to this job.
source "$script_dir/envs.sh"
python -m pip install setuptools wheel ninja
if [[ ! -e "$flash_kda_root" ]]; then
    git clone https://github.com/MoonshotAI/FlashKDA.git "$flash_kda_root"
fi
(
    cd -- "$flash_kda_root"
    git submodule update --init --recursive
    python - <<'PYTHON'
import os
import torch

target = os.environ["FLASH_KDA_CUDA_ARCHS"]
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = f"{gpu.major}{gpu.minor}a" if target.lower() == "auto" else target
    print(f"FlashKDA build: {gpu.name}, compute capability {gpu.major}.{gpu.minor}, target {arch}")
elif target.lower() == "auto":
    raise SystemExit("Run the build on the allocated GPU node or set FLASH_KDA_CUDA_ARCHS for a cross-build.")
else:
    print(f"FlashKDA cross-build: target {target}")
PYTHON
    python -m pip install -v --no-build-isolation .
)

# Check the installed toolchain and CUDA extension without running GPU kernels.
python --version
nvcc --version
gcc -dumpfullversion
numactl --version
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)'
python -c 'import flash_kda; print("FlashKDA:", flash_kda.__file__)'
printf 'Build complete. Activate with: source %q/envs.sh\n' "$script_dir"
