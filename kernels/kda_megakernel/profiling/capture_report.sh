#!/bin/bash
# Run on the allocated GPU after sourcing tools/env.sh, from the repository root.
set -eo pipefail
exp=${1:?Usage: capture_report.sh EXPERIMENT_DIR}
export KDA_EXPERIMENT="${exp##*/}"
# Current baseline captures have their own collection alongside the historical suite.
folder=ncu-single-conv
python - "$exp/reports/$folder" <<'PY'
import json
import sys
from pathlib import Path
from kernels.kda_megakernel.profiling.device import device_identity

gpu = device_identity()
print(f"Capturing on {gpu['device']} (UUID {gpu['uuid']})")
for path in Path(sys.argv[1]).glob('*/identity.json'):
    record = json.loads(path.read_text())
    if record.get('metadata', record)['uuid'] != gpu['uuid']:
        raise SystemExit(f'{path} belongs to a different GPU. Select a separate experiment.')
PY
metrics=$(python -m kernels.kda_megakernel.profiling.counters)
cases=$(python -c 'from kernels.kda_megakernel.cases.inputs import load_cases;print(" ".join(c["id"] for c in sorted(load_cases(),key=lambda c:c["total_tokens"]*c["hv"])))')
for cid in $cases; do
    dest="$exp/reports/$folder/$cid"
    if [[ -s "$dest/owners.csv" && -s "$dest/counters.csv" && -s "$dest/identity.json" ]]; then continue; fi
    mkdir -p "$dest"
    replay=kernel
    replay_options=()
    case "$cid" in
        *_01|*_05|*_06|*_10|*_16|*_20) replay=application; replay_options=(--app-replay-mode strict) ;;
    esac
    echo "$replay" > "$dest/replay-mode.txt"
    ncu --devices 0 --replay-mode "$replay" "${replay_options[@]}" --nvtx --profile-from-start off --metrics "$metrics" -f -o "$dest/capture" python -m "kernels.kda_megakernel.${KDA_EXPERIMENT}.reports.single_conv" ncu "$cid" > "$dest/capture.log" 2>&1
    ncu --import "$dest/capture.ncu-rep" --page raw --csv --print-units base > "$dest/counters.csv"
    ncu --import "$dest/capture.ncu-rep" --page raw --csv --print-units base --print-nvtx-rename kernel > "$dest/owners.csv"
done
