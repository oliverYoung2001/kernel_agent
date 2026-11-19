#!/usr/bin/env bash
# Source this file to activate the environment used by LACP/scripts/envs.sh.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Activate with: source %s\n' "$0" >&2
    exit 1
fi

_kernel_activate_environment() {
    local conda_root="${KERNEL_CONDA_ROOT:-$HOME/anaconda3}"
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

    # Preserve the package versions and activation order of the source script.
    spack load numactl@2.0.18 || return
    conda deactivate || return
    conda deactivate || return
    conda activate linear || return

    export CUDA_HOME="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
    export CPATH="$CUDA_HOME/include/cccl:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}"

    spack load gcc@13.4.0 || return
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
    unset -f _kernel_prepare_environment _kernel_activate_environment
else
    unset -f _kernel_prepare_environment _kernel_activate_environment
    return 1
fi
