#!/usr/bin/env bash
# Source on the compute node so hostname and the activated tools refer to the same job.
_kda_activate_node_env() {
    local -
    set +u
    local repo_root
    repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P) || return

    source "$repo_root/skills/prepare-environment/envs.sh" || return
    export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
}

if _kda_activate_node_env; then
    unset -f _kda_activate_node_env
else
    unset -f _kda_activate_node_env
    return 1 2>/dev/null || exit 1
fi
