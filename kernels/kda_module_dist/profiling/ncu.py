"""Explicit NCU range boundaries around one warmed complete baseline call."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from functools import wraps
import os

import torch
import torch.distributed as dist

from ..cases.inputs import make_inputs_from_recipe
from ..run import IMPLEMENTATIONS
from .evidence import compiled_binaries, identity, write
from .inputs import materialize


@contextmanager
def collective_nvtx(control_group):
    """Label collective launches only for NCU lockstep; leave ordinary runs intact."""
    originals = {}
    for name in ("all_gather_into_tensor", "all_to_all_single", "reduce_scatter_tensor"):
        original = getattr(dist, name)
        originals[name] = original

        @wraps(original)
        def wrapped(*args, _operation=original, **kwargs):
            # Keep excluded NCCL off the GPUs until every rank finishes replay.
            dist.barrier(group=control_group)
            torch.cuda.nvtx.range_push("kda_collective")
            try:
                result = _operation(*args, **kwargs)
                # Drain communication before the next per-kernel profiling region.
                # NCU cannot acquire counters while an excluded NCCL kernel spins.
                torch.cuda.synchronize()
                return result
            finally:
                torch.cuda.nvtx.range_pop()
        setattr(dist, name, wrapped)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(dist, name, original)


def capture_one(args, cases):
    if len(cases) != 1 or len(args.impl) != 1:
        raise ValueError("NCU capture requires one case and one implementation")
    case, name = cases[0], args.impl[0]
    arguments = materialize(case, implementation=name)
    for _ in range(args.warmup):
        IMPLEMENTATIONS[name](**arguments)
    torch.cuda.synchronize()
    dist.barrier()
    control_group = dist.new_group(backend="gloo")
    torch.cuda.nvtx.range_push(f"kda_dist/{name}/{case['id']}")
    torch.cuda.cudart().cudaProfilerStart()
    try:
        with collective_nvtx(control_group):
            IMPLEMENTATIONS[name](**arguments)
            torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
    folder = Path(args.experiment) / "baselines" / name / "profiling/ncu" / case["id"]
    filter_mode = os.environ.get("KDA_NCU_FILTER_MODE", "all")
    if filter_mode != "all":
        folder = folder / filter_mode
    write(folder / f"rank-{dist.get_rank():04d}.identity.json",
          dict(metadata=identity(args.manifest), binaries=compiled_binaries(), case=case,
               implementation=name, filter_mode=filter_mode,
               profiling_collective_synchronize=True,
               profiling_collective_cpu_barrier=True,
               boundary="one complete call between cudaProfilerStart/Stop"))
    dist.barrier()
