#!/usr/bin/env bash
# Source this file to activate the environment used by LACP/scripts/envs.sh.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Activate with: source %s\n' "$0" >&2
    exit 1
fi

_kernel_activate_cluster() {
    # Preserve the current cluster's package versions and activation order.
    spack load numactl@2.0.18 || return
    conda deactivate || return
    conda deactivate || return
    conda activate linear || return
    export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
    if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
        printf 'CUDA compiler not found: %s/bin/nvcc\n' "$CUDA_HOME" >&2
        return 1
    fi
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include/cccl:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}"
    spack load gcc@13.4.0 || return
}

_kernel_activate_environment() {
    local node
    node=$(hostname) || return
    local conda_default="$HOME/anaconda3"
    case "${node,,}" in
        *fuse*) conda_default="$HOME/.local/miniconda3" ;;
    esac
    local conda_root="${KERNEL_CONDA_ROOT:-$conda_default}"
    local spack_root="${KERNEL_SPACK_ROOT:-$HOME/.local/spack}"
    local setup_file

    for setup_file in \
        "$conda_root/etc/profile.d/conda.sh" \
        "$spack_root/share/spack/setup-env.sh"; do
        if [[ ! -f "$setup_file" ]]; then
            printf 'Environment setup file not found: %s\n' "$setup_file" >&2
            return 1
        fi
    done

    # Initialize both shell integrations instead of relying on .bashrc.
    source "$conda_root/etc/profile.d/conda.sh" || return
    source "$spack_root/share/spack/setup-env.sh" || return

    case "${node,,}" in
        *fuse*)
            conda activate linear || return
            spack load gcc@12.2.0 || return
            if [[ -x /usr/local/cuda-13.2/bin/nvcc ]]; then
                export CUDA_HOME=/usr/local/cuda-13.2
                export PATH="$CUDA_HOME/bin:$PATH"
            fi
            export OMP_NUM_THREADS=8
            ;;
        *qcjz*|*qhjz*|*bjdb*)
            _kernel_activate_cluster || return
            ;;
        *)
            # Keep the original portable defaults for explicitly configured hosts.
            _kernel_activate_cluster || return
            ;;
    esac
}

_kernel_prepare_environment() {
    local restore_nounset=0
    local activation_status=0

    # Conda activation hooks may read unset compiler variables under set -u.
    if [[ $- == *u* ]]; then
        restore_nounset=1
        set +u
    fi
    if _kernel_activate_environment; then
        activation_status=0
    else
        activation_status=$?
    fi
    if (( restore_nounset )); then
        set -u
    fi
    return "$activation_status"
}

if _kernel_prepare_environment; then
    unset -f _kernel_prepare_environment _kernel_activate_environment _kernel_activate_cluster
else
    unset -f _kernel_prepare_environment _kernel_activate_environment _kernel_activate_cluster
    return 1
fi
