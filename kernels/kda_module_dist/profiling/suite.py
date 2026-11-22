"""Collect formal baseline verification, ordinary timing and all-rank activities."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.distributed as dist

from ..cases.inputs import make_inputs_from_recipe
from ..run import IMPLEMENTATIONS, metrics
from .cupti import capture
from .evidence import BASELINES, command, identity, write
from .inputs import materialize


def finite_output(value):
    finite = torch.isfinite(value).all().to(torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def errors(error):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, error)
    return [dict(rank=rank, error=value) for rank, value in enumerate(gathered) if value]


def timing(operation, arguments, warmup, repeat):
    for _ in range(warmup):
        operation(**arguments)
    torch.cuda.synchronize()
    dist.barrier()
    local = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter_ns()
        operation(**arguments)
        torch.cuda.synchronize()
        local.append((time.perf_counter_ns() - start) / 1e6)
    values = torch.tensor(local, device=arguments["inpx"].device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    samples = values.tolist()
    mean = statistics.mean(samples)
    return dict(samples_ms=samples, local_samples_ms=local, mean_ms=mean,
                stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
                cv_pct=100 * statistics.stdev(samples) / mean if len(samples) > 1 else 0.0,
                warmup=warmup, repeat=repeat,
                aggregation="each sample is the maximum complete-call wall time across ranks; mean of these samples",
                boundary="Python entry to all local outputs ready; CUDA synchronize at end; setup and barriers excluded; profiler disabled")


def collect(args, cases):
    experiment = Path(args.experiment).resolve()
    rank, p = dist.get_rank(), dist.get_world_size()
    if any(case["parallel_size"] != p for case in cases):
        raise ValueError("All suite cases must match WORLD_SIZE")
    if any(name not in BASELINES for name in args.impl):
        raise ValueError("The current report suite contains only tp, ulysses and kcp")
    metadata = identity(args.manifest)
    if rank == 0:
        print(f"KDA_DIST_SUITE_START world_size={p} cases={len(cases)}", flush=True)
    for case_index, case in enumerate(cases):
        cid = case["id"]
        if args.resume:
            cached = []
            for name in args.impl:
                path = experiment / "baselines" / name / "records" / f"{cid}.json"
                cached.append(json.loads(path.read_text()) if path.exists() else {})
            if any(item.get('status') == 'complete' and item.get('source_identity') != metadata['source_sha256'] for item in cached):
                raise FileExistsError(f"Preserving completed evidence for {cid}: collector/source identity changed; choose a fresh --experiment directory")
            if all(item.get('status') == 'complete' and item.get('source_identity') == metadata['source_sha256']
                   and item.get('timing', {}).get('warmup') == args.warmup
                   and item.get('timing', {}).get('repeat') == args.repeat for item in cached):
                if rank == 0:
                    print(f"KDA_DIST_CASE_RESUME_SKIP {cid}", flush=True)
                continue
        identity_path = experiment / f"environment/p{p}/{cid}/rank-{rank:04d}.json"
        write(identity_path, metadata)
        expected = None
        truth_error = None
        truth = None
        try:
            truth = materialize(case, implementation="tp", rank=rank)
            IMPLEMENTATIONS["tp"](**truth)
            if not finite_output(truth["out"]):
                raise ValueError("TP ground truth contains nonfinite output")
            expected = truth["out"].cpu()
        except Exception:
            truth_error = traceback.format_exc()
        finally:
            del truth
            gc.collect()
            torch.cuda.empty_cache()
        truth_errors = errors(truth_error)
        names = list(args.impl)
        shift = case["seed"] % len(names)
        names = names[shift:] + names[:shift]
        for name in names:
            folder = experiment / "baselines" / name
            target = folder / "records" / f"{cid}.json"
            skip = args.resume and target.exists()
            skip_tensor = torch.tensor(int(skip), device="cuda")
            dist.broadcast(skip_tensor, src=0)
            if bool(skip_tensor.item()):
                saved = json.loads(target.read_text())
                if saved.get("status") == "complete" and saved.get("source_identity") == metadata["source_sha256"]:
                    continue
            record = dict(case=case, implementation=name, baseline=True, versions=[],
                          world_size=p, source_identity=metadata["source_sha256"],
                          metadata_path=str(identity_path),
                          ground_truth="tp", threshold=args.threshold, status="pending")
            arguments = None
            local_error = None
            if rank == 0:
                print(f"KDA_DIST_CASE_START {cid} {name}", flush=True)
            try:
                if truth_errors:
                    record["ground_truth_errors"] = truth_errors
                    raise RuntimeError("TP ground truth failed")
                arguments = materialize(case, implementation=name, rank=rank)
                IMPLEMENTATIONS[name](**arguments)
                verification = metrics(arguments["out"], expected)
                verification.update(ground_truth="tp", threshold=args.threshold,
                                    independent_mathematical_oracle=False,
                                    role="ground_truth_reexecution" if name == "tp" else "candidate")
                verification["pass"] = verification["finite"] and verification["scaled_max"] <= args.threshold and verification["rmse_ratio"] <= args.threshold
                record["verification"] = verification
                before = command(["nvidia-smi", "--id=" + metadata["uuid"],
                                  "--query-gpu=uuid,clocks.sm,clocks.mem,power.limit,power.draw,temperature.gpu", "--format=csv,noheader"])
                record["timing"] = timing(IMPLEMENTATIONS[name], arguments, args.warmup, args.repeat)
                after = command(["nvidia-smi", "--id=" + metadata["uuid"],
                                 "--query-gpu=uuid,clocks.sm,clocks.mem,power.limit,power.draw,temperature.gpu", "--format=csv,noheader"])
                record["timing"]["device_before"] = before
                record["timing"]["device_after"] = after
                record["timing"]["finite"] = finite_output(arguments["out"])
                write(folder / "end_to_end" / f"{cid}.rank-{rank:04d}.json", record["timing"])
                write(folder / "verification" / f"{cid}.rank-{rank:04d}.json", verification)
                record["local_profile"] = capture(folder / "profiling/cupti", case, IMPLEMENTATIONS[name], arguments, metadata)
                record["status"] = "complete"
            except Exception:
                local_error = traceback.format_exc()
                record["status"] = "error"
            finally:
                del arguments
                gc.collect()
                torch.cuda.empty_cache()
            failures = errors(local_error)
            if failures:
                record["status"] = "error"
                record["errors"] = failures
            profiles = [None] * p
            dist.all_gather_object(profiles, record.get("local_profile"))
            record["profiles"] = profiles
            if rank == 0:
                write(target, record)
                print(f"KDA_DIST_CASE_DONE {cid} {name} {record['status']} verification={record.get('verification', {}).get('pass')}", flush=True)
        del expected
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        for name in args.impl:
            folder = experiment / "baselines" / name
            records = [json.loads(path.read_text()) for path in sorted((folder / "records").glob("*.json"))]
            write(folder / "end_to_end.json", {record["case"]["id"]: record.get("timing") for record in records})
            write(folder / "verification.json", {record["case"]["id"]: record.get("verification") for record in records})
        print(f"KDA_DIST_SUITE_COMPLETE world_size={p}", flush=True)


def profile_only(args, cases):
    metadata = identity(args.manifest)
    for case in cases:
        for name in args.impl:
            arguments = make_inputs_from_recipe(case, implementation=name)
            for _ in range(args.warmup):
                IMPLEMENTATIONS[name](**arguments)
            capture(Path(args.experiment) / "baselines" / name / "profiling/cupti",
                    case, IMPLEMENTATIONS[name], arguments, metadata)
            del arguments
            torch.cuda.empty_cache()
