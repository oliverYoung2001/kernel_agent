"""Regenerate the twenty model-shaped recipes without downloading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


# KDA dimensions, not the separate MLA/DSA dimensions in the same checkpoints.
# Configurations were read from the publisher repositories on 2026-11-20.
MODELS = (
    ("kimi_k31_pro", "Kimi K3.1 Pro", "moonshotai/Kimi-K3.1-Pro",
     "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e", 7168, 96),
    ("glm53_flash", "GLM-5.3-Flash", "zai-org/GLM-5.3-Flash",
     "46a050bec13002ad29ee795c16e61cf4fa3c4a9c", 4096, 64),
    ("kimi_linear_48b_a3b", "Kimi Linear 48B-A3B", "moonshotai/Kimi-Linear-48B-A3B-Instruct",
     "e1df551a447157d4658b573f9a695d57658590e9", 2304, 32),
    ("ling31_flash", "Ling-3.1-flash", "inclusionAI/Ling-3.1-flash",
     "dde57b1c6a365aa0e0a33e15cecce0f2be96f5a3", 4096, 64),
)
LENGTH_SEED = 20261120
TOKEN_BUDGET = 128 * 1024


def build_manifest() -> dict:
    """Use the same seeded random partition for comparable model workloads."""
    cuts = [0, *sorted(random.Random(LENGTH_SEED).sample(range(1, TOKEN_BUDGET), 4)), TOKEN_BUDGET]
    lengths = [end - start for start, end in zip(cuts, cuts[1:])]
    manifest = {
        "schema_version": 3,
        "description": "GLM-style full attention with four model geometries, each with one packed batch and four fixed lengths",
        "tokens_per_k": 1024,
        "variable_length_seed": LENGTH_SEED,
        "variable_length_total_tokens": TOKEN_BUDGET,
        "defaults": {
            "batch": 1, "variable_length": False, "conv_width": 4,
            "dtype": "bfloat16", "large": True,
        },
        "models": {},
        "cases": [],
    }
    for key, name, repo, revision, hidden, heads in MODELS:
        shape = {"hidden": hidden, "hqk": heads, "hv": heads, "dk": 128, "dv": 128,
                 "conv_width": 4}
        manifest["models"][key] = {
            "name": name, "repository": repo, "revision": revision,
            "config_url": f"https://huggingface.co/{repo}/blob/{revision}/config.json",
            **shape,
        }
        for sequence in (None, 16 * 1024, 32 * 1024, 64 * 1024, TOKEN_BUDGET):
            number = len(manifest["cases"]) + 1
            case = {"id": f"kda_{key}_{number:02d}", "model": key, **shape,
                    "seed": 41 + number}
            if sequence is None:
                case.update(batch=5, variable_length=True, sequence=TOKEN_BUDGET,
                            sequence_lengths=list(lengths), total_tokens=TOKEN_BUDGET,
                            purpose=f"{name}: five independent packed sequences, total 128K tokens")
            else:
                case.update(batch=1, variable_length=False, sequence=sequence, total_tokens=sequence,
                            purpose=f"{name}: batch one, {sequence // 1024}K tokens")
            manifest["cases"].append(case)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--check", action="store_true", help="Check that the committed manifest is reproducible")
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
