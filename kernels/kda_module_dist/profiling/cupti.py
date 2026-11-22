"""Capture every rank's complete call using torch.profiler CUDA activities."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch
import torch.distributed as dist

from .evidence import compiled_binaries, write


def extract_kernels(events):
    runtime = {}
    for event in events:
        if event.get("cat") in ("cuda_runtime", "cuda_driver"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                runtime[correlation] = event["ts"]
    kernels = []
    for original_index, event in enumerate(events):
        if event.get("cat") != "kernel":
            continue
        args = event.get("args", {})
        correlation = args.get("correlation")
        kernels.append(dict(name=event["name"], duration_us=event["dur"],
                            start_us=event["ts"], launch_host_us=runtime.get(correlation),
                            trace_event_index=original_index, correlation=correlation,
                            count=1, args=args))
    kernels.sort(key=lambda item: (item["launch_host_us"] if item["launch_host_us"] is not None else item["start_us"],
                                   item["trace_event_index"]))
    for index, event in enumerate(kernels, 1):
        event["launch_index"] = index
    return kernels


def capture(folder, case, operation, arguments, metadata):
    folder = Path(folder) / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    temporary = folder / f"rank-{rank:04d}.trace.json"
    trace = temporary.with_suffix(temporary.suffix + ".gz")
    torch.cuda.synchronize()
    dist.barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, with_stack=False) as profiler:
        operation(**arguments)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(temporary))
    raw = temporary.read_bytes()
    with gzip.open(trace, "wb") as target:
        target.write(raw)
    temporary.unlink()
    parsed = json.loads(raw)
    kernels = extract_kernels(parsed["traceEvents"])
    if not kernels:
        raise RuntimeError("CUPTI did not record any GPU kernel")
    record = dict(metadata=metadata, case=case, trace=str(trace), kernels=kernels,
                  binaries=compiled_binaries(),
                  boundary="one complete unmodified baseline call; synchronize inside profile; barriers outside profile",
                  launch_order="CUDA runtime correlation timestamp, falling back to GPU start timestamp",
                  kernel_count=len(kernels), device_time_sum_us=sum(item["duration_us"] for item in kernels),
                  counts_reason="CUPTI activity does not measure executed FLOPs or DRAM/shared traffic")
    write(folder / f"rank-{rank:04d}.kernels.json", record)
    dist.barrier()
    return dict(trace=str(trace), parsed=str(folder / f"rank-{rank:04d}.kernels.json"),
                kernel_count=len(kernels), device_time_sum_us=record["device_time_sum_us"])
