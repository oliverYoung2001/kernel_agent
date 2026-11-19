#!/usr/bin/env bash
# Build the Conda and Spack environment used by LACP/scripts/envs.sh.
set -eo pipefail
set +u

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_root="${KERNEL_CONDA_ROOT:-$HOME/anaconda3}"
spack_root="${KERNEL_SPACK_ROOT:-$HOME/.local/spack}"

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

# Activate the completed environment and check the installed toolchain.
source "$script_dir/envs.sh"
python --version
nvcc --version
gcc -dumpfullversion
numactl --version
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)'
printf 'Build complete. Activate with: source %q/envs.sh\n' "$script_dir"
