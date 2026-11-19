"""List, validate, or execute the standalone KDA reference cases."""

from __future__ import annotations

import argparse
import json

import torch

from .inputs import CASES, make_inputs
from .reference import reference, validate_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List all twenty input recipes")
    parser.add_argument("--validate-inputs", action="store_true", help="Validate all twenty shapes using meta tensors")
    parser.add_argument("--case", choices=[c["id"] for c in CASES] + ["all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-large", action="store_true", help="Allow allocating and running kda_20")
    args = parser.parse_args()
    if args.list or not (args.case or args.validate_inputs):
        print(json.dumps(CASES, indent=2))
    if args.validate_inputs:
        for case in CASES:
            validate_arguments(**make_inputs(case["id"], device="meta"))
        print(json.dumps({"input_specs_validated": 20, "numerical_execution": False}))
    if args.case:
        selected = CASES if args.case == "all" else [c for c in CASES if c["id"] == args.case]
        if any(c["large"] for c in selected) and not args.include_large:
            parser.error("kda_20 requires --include-large; --case all includes this case")
        for case in selected:
            arguments = make_inputs(case["id"], device=args.device)
            reference(**arguments)
            finite = all(bool(torch.isfinite(arguments[name]).all()) for name in ("out", "final_state"))
            print(json.dumps({"case": case["id"], "reference_completed": True,
                              "output_shape": list(arguments["out"].shape), "finite": finite,
                              "candidate_verified": False}))
            if not finite:
                raise RuntimeError(f"Non-finite output: {case['id']}")


if __name__ == "__main__":
    main()
