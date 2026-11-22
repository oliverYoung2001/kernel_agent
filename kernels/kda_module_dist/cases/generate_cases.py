"""Reproduce four model geometries with two 8-way and three 16-way recipes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kernels.kda_megakernel.cases.generate_cases import MODELS

LENGTHS = [50671, 7451, 9923, 8622, 54405]


def build_manifest():
    manifest = {
        "schema_version": 1,
        "description": "Distributed complete KDA module; contiguous naive token partition; output only",
        "tokens_per_k": 1024,
        "defaults": {
            "batch": 1, "variable_length": False, "conv_width": 4,
            "dtype": "bfloat16", "large": True,
            "layout": "contiguous [total_tokens / parallel_size, hidden]; rank order follows flattened batch*sequence",
            "weight_layout": "tp: contiguous head shards; ulysses/kcp/reference: replicated",
            "random_algorithm": "cpu_torch_randn_row_blocks_v1; block_rows=128; seed=case_seed+1000003*stream+block",
            "values": "inpx N(0,1); projections N(0,fan_in^-1/2); conv N(0,0.2); norms=1; A_log=dt_bias=0; zero initial state; out=NaN",
        },
        "models": {}, "cases": [],
    }
    for key, name, repo, revision, hidden, heads in MODELS:
        shape = dict(hidden=hidden, hqk=heads, hv=heads, dk=128, dv=128, conv_width=4)
        manifest["models"][key] = dict(name=name, repository=repo, revision=revision,
                                      config_url=f"https://huggingface.co/{repo}/blob/{revision}/config.json", **shape)
        for p, variable, sequence in ((8, True, 131072), (8, False, 131072),
                                      (16, True, 131072), (16, False, 131072), (16, False, 262144)):
            number = len(manifest["cases"]) + 1
            case = dict(id=f"kda_dist_{key}_{number:02d}", model=key, **shape,
                        seed=41 + number, parallel_size=p, batch=5 if variable else 1,
                        variable_length=variable, sequence=sequence, total_tokens=sequence,
                        purpose=f"{name}: {p} ranks, " + ("five packed sequences, total 128K" if variable else f"batch one, {sequence // 1024}K"))
            if variable:
                case["sequence_lengths"] = list(LENGTHS)
            manifest["cases"].append(case)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if args.output.read_text() != content:
            raise SystemExit(f"Manifest differs: {args.output}")
        print(f"Reproducible manifest: {args.output}")
    else:
        args.output.write_text(content)
        print(f"Wrote 20 cases: {args.output}")


if __name__ == "__main__":
    main()
