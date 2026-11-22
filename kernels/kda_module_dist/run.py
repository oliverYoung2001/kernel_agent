"""List, validate, execute, verify and time distributed complete KDA workloads."""
from __future__ import annotations

import argparse
from datetime import timedelta
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.kda_module_dist.baseline import baseline_kcp, baseline_tp, baseline_ulysses
from kernels.kda_module_dist.cases.inputs import MANIFEST_PATH, load_cases, make_inputs_from_recipe
from kernels.kda_module_dist.distributed import WEIGHT_NAMES, topology, validate_input_specs
from kernels.kda_module_dist.reference import reference

IMPLEMENTATIONS = dict(reference=reference, tp=baseline_tp, ulysses=baseline_ulysses, kcp=baseline_kcp)


def metrics(actual, expected, group=None):
    """Reduce FP64 error statistics across every element on every rank."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Output shape or dtype differs from the reference")
    maximum = torch.zeros(2, device=actual.device, dtype=torch.float64)
    energy = torch.zeros(2, device=actual.device, dtype=torch.float64)
    finite = torch.ones((), device=actual.device, dtype=torch.int32)
    for a, b in zip(actual.flatten().split(1048576), expected.flatten().split(1048576)):
        a, b = a.double(), b.to(device=actual.device, dtype=torch.float64)
        finite.mul_((torch.isfinite(a).all() & torch.isfinite(b).all()).to(torch.int32))
        delta = a - b
        maximum = torch.maximum(maximum, torch.stack((delta.abs().max(), b.abs().max())))
        energy += torch.stack((delta.square().sum(), b.square().sum()))
    if topology(group)[0] > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        dist.all_reduce(energy, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    scaled_max = (maximum[0] / maximum[1].clamp_min(1e-12)).item()
    rmse_ratio = (energy[0] / energy[1].clamp_min(1e-24)).sqrt().item()
    return dict(finite=bool(finite.item()), scaled_max=scaled_max, rmse_ratio=rmse_ratio)


def resolve(name):
    if name in IMPLEMENTATIONS:
        return IMPLEMENTATIONS[name]
    # A candidate is an importable module:function with the same explicit signature.
    module, sep, function = name.partition(":")
    if not sep:
        raise ValueError(f"Unknown implementation {name}; use tp, ulysses, kcp, reference or module:function")
    return getattr(importlib.import_module(module), function)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cases", "--case", nargs="+")
    parser.add_argument("--impl", nargs="+", default=["tp", "ulysses", "kcp"])
    parser.add_argument("--weight-layout", choices=["tp", "ulysses", "kcp"], default="tp",
                        help="Weight ABI for module:function candidates")
    parser.add_argument("--mode", choices=["execute", "verify", "timing", "suite", "profile", "capture"], default="execute")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--include-large", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment", type=Path, default=Path(__file__).with_name("experiment0"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ground-truth", choices=["tp", "reference"], default="tp")
    parser.add_argument("--cp-alignment", action="store_true", help="Verify the native-compatible CP API against FLA's CP test grid")
    parser.add_argument("--fla-root", type=Path, default=Path.home() / "yhy/llm/flash-linear-attention")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat < 1 or args.threshold <= 0:
        parser.error("warmup >= 0, repeat >= 1 and threshold > 0 are required")
    if args.cp_alignment:
        if args.mode != "verify":
            parser.error("--cp-alignment requires --mode verify")
        from kernels.kda_module_dist.verification.cp_alignment import run
        return run(args.output or args.experiment / "verification/kcp_alignment", args.cases,
                   args.resume, args.fla_root, args.list)
    cases = load_cases(args.manifest)
    if args.cases and args.cases != ["all"]:
        unknown = set(args.cases) - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = tuple(case for case in cases if case["id"] in args.cases)
    if args.list:
        print(json.dumps(cases, indent=2, ensure_ascii=False))
        if not args.validate_inputs:
            return
    operations = {name: resolve(name) for name in args.impl}
    if args.validate_inputs:
        count = 0
        for case in cases:
            for rank in range(case["parallel_size"]):
                for name, operation in operations.items():
                    layout = name if name in IMPLEMENTATIONS else args.weight_layout
                    arguments = make_inputs_from_recipe(case, implementation=layout, device="meta", rank=rank)
                    inspect.signature(operation).bind(**arguments)
                    validate_input_specs(arguments["inpx"], arguments["out"],
                                         cu_seqlens=arguments["cu_seqlens"], hq=case["hqk"], hv=case["hv"],
                                         dk=case["dk"], dv=case["dv"], weights={key: arguments[key] for key in WEIGHT_NAMES},
                                         implementation=layout, world_size=case["parallel_size"])
                    count += 1
        print(f"Validated {len(cases)} recipes, {count} rank/implementation signatures and tensor specifications; numerical execution not performed")
        return
    if not args.include_large and any(case.get("large", False) for case in cases):
        parser.error("Production cases require --include-large and are allocated on demand")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
    p, rank = topology()
    results = []
    failed = False
    try:
        if any(case["parallel_size"] != p for case in cases):
            parser.error(f"WORLD_SIZE={p}; select only cases with parallel_size={p}")
        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.mode == "suite":
            from kernels.kda_module_dist.profiling.suite import collect
            return collect(args, cases)
        if args.mode == "profile":
            from kernels.kda_module_dist.profiling.suite import profile_only
            return profile_only(args, cases)
        if args.mode == "capture":
            from kernels.kda_module_dist.profiling.ncu import capture_one
            return capture_one(args, cases)
        for case in cases:
            expected = None
            if args.mode == "verify":
                truth = make_inputs_from_recipe(case, implementation=args.ground_truth, rank=rank)
                IMPLEMENTATIONS[args.ground_truth](**truth)
                expected = truth["out"].cpu()
                del truth
            for name, operation in operations.items():
                layout = name if name in IMPLEMENTATIONS else args.weight_layout
                arguments = make_inputs_from_recipe(case, implementation=layout, rank=rank)
                record = dict(case=case["id"], implementation=name, mode=args.mode, parallel_size=p,
                              gpu=torch.cuda.get_device_name(), torch=torch.__version__)
                if args.mode == "timing":
                    for _ in range(args.warmup):
                        operation(**arguments)
                    samples = []
                    for _ in range(args.repeat):
                        torch.cuda.synchronize()
                        dist.barrier()
                        start = time.perf_counter_ns()
                        operation(**arguments)
                        torch.cuda.synchronize()
                        elapsed = torch.tensor((time.perf_counter_ns() - start) / 1e6, device=arguments["inpx"].device, dtype=torch.float64)
                        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                        samples.append(elapsed.item())
                    record.update(warmup=args.warmup, repeat=args.repeat, samples_ms=samples,
                                  mean_ms=sum(samples) / len(samples))
                else:
                    operation(**arguments)
                finite = torch.isfinite(arguments["out"]).all().to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                record["finite"] = bool(finite.item())
                if expected is not None:
                    record["ground_truth"] = args.ground_truth
                    record.update(metrics(arguments["out"], expected))
                    record["pass"] = record["finite"] and record["scaled_max"] <= args.threshold and record["rmse_ratio"] <= args.threshold
                    record["threshold"] = args.threshold
                else:
                    record["pass"] = record["finite"]
                failed |= not record["pass"]
                results.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
                del arguments
        if rank == 0 and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        if failed:
            raise SystemExit(1)
    finally:
        if owned_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
