#!/usr/bin/env bash
#SBATCH --job-name=kcp-alignment
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=00:59:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source kernels/kda_module_dist/tools/env.sh
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export FLA_CI_ENV=0
python - <<'PY' > "kernels/kda_module_dist/experiment1/verification/gpu-environment-${SLURM_JOB_ID}.txt"
import json
import os
import socket

import torch
import triton

print(json.dumps(dict(hostname=socket.gethostname(), job=os.environ["SLURM_JOB_ID"],
                     torch=torch.__version__, cuda_build=torch.version.cuda,
                     triton=triton.__version__, nccl=torch.cuda.nccl.version())))
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
    >> "kernels/kda_module_dist/experiment1/verification/gpu-environment-${SLURM_JOB_ID}.txt"
python -m kernels.kda_module_dist.run --mode verify --cp-alignment --resume \
    --experiment kernels/kda_module_dist/experiment1
# Recheck complete-module integration, including packed GVA and weak decay.
torchrun --standalone --nproc_per_node=2 \
    -m kernels.kda_module_dist.verification.check --gpu
