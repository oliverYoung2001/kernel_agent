#!/usr/bin/env bash
# Source on the compute node so hostname and the activated tools refer to the same job.
_kda_activate_node_env() {
    local -
    set +u
    local node repo_root conda_root spack_root
    node=$(hostname) || return
    repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P) || return

    case "${node,,}" in
        *fuse*)
            conda_root=${KERNEL_CONDA_ROOT:-$HOME/.local/miniconda3}
            spack_root=${KERNEL_SPACK_ROOT:-$HOME/.local/spack}
            source "$conda_root/etc/profile.d/conda.sh" || return
            source "$spack_root/share/spack/setup-env.sh" || return
            conda activate linear || return
            spack load gcc@12.2.0 || return
            if [[ -x /usr/local/cuda-13.2/bin/nvcc ]]; then
                export CUDA_HOME=/usr/local/cuda-13.2
                export PATH="$CUDA_HOME/bin:$PATH"
            fi
            export OMP_NUM_THREADS=8
            ;;
        *qcjz*|*bjdb*)
            source "$repo_root/skills/prepare-environment/envs.sh" || return
            ;;
    esac
    export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
}

if _kda_activate_node_env; then
    unset -f _kda_activate_node_env
else
    unset -f _kda_activate_node_env
    return 1 2>/dev/null || exit 1
fi
