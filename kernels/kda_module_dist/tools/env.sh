#!/usr/bin/env bash
# Reuse the repository's node-aware activation and Python path setup.
_kda_dist_activate() {
    local repo_root
    repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P) || return
    source "$repo_root/skills/prepare-environment/envs.sh" || return
    export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
}
if _kda_dist_activate; then
    unset -f _kda_dist_activate
else
    unset -f _kda_dist_activate
    return 1 2>/dev/null || exit 1
fi
