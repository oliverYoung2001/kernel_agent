#!/usr/bin/env bash
# Run one torchrun launcher per allocated Slurm node.
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$script_dir/env.sh"
repo_root=$(cd -- "$script_dir/../../.." && pwd -P)
cd -- "$repo_root"
if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_NODEID:-} ]]; then
    printf 'Run under srun with one launcher task per node.\n' >&2
    exit 1
fi
master_addr=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_STEP_NODELIST:-$SLURM_JOB_NODELIST}" | head -n 1)}
master_port=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
exec torchrun --nnodes="$SLURM_NNODES" --nproc_per_node="${KERNEL_GPUS_PER_NODE:-8}" \
    --node_rank="$SLURM_NODEID" --master_addr="$master_addr" --master_port="$master_port" \
    -m kernels.kda_module_dist.run "$@"
