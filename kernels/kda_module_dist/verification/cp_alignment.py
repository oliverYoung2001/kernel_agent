"""Compare the exported CP API, upstream CP and its independent recurrent oracle."""
from __future__ import annotations

from datetime import timedelta
from contextlib import nullcontext
import faulthandler
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLA = Path.home() / "yhy/llm/flash-linear-attention"
NAMES = ("out", "dq", "dk", "dv", "dg", "dbeta")


def upstream(root):
    # Use the requested checkout, including its test inputs and parameter grid.
    sys.path.insert(0, str(root))
    module = importlib.import_module("tests.cp.test_kda")
    assert Path(module.__file__).resolve() == root / "tests/cp/test_kda.py"
    import fla
    assert Path(fla.__file__).resolve().is_relative_to(root)
    return module


def cases(module):
    mark = next(mark for mark in module.test_chunk.pytestmark if mark.name == "parametrize")
    names, parameters = mark.args
    return [dict(id=parameter.id, **dict(zip(names, parameter.values))) for parameter in parameters]


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def errors(actual, expected):
    """FP64 statistics over every element; never hide non-finite values."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Output shape or dtype mismatch")
    a, b = actual.double(), expected.double()
    finite = bool((a.isfinite().all() & b.isfinite().all()).item())
    if not finite:
        return dict(finite=False, exact=False, max_abs=None, rmse_ratio=None)
    delta = a - b
    return dict(finite=True, exact=bool(torch.equal(actual, expected)),
                max_abs=delta.abs().max().item(),
                rmse_ratio=(delta.square().mean().sqrt() / (b.square().mean().sqrt() + 1e-8)).item())


def call(operation, tensors, do, options, adapter=False):
    leaves = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    q, k, v, g, beta = leaves
    if adapter:
        # Match the noncontiguous QKV channel views returned by CP convolution.
        heads, width = q.shape[-2:]
        packed = torch.cat([tensor.flatten(-2) for tensor in (q, k, v)], dim=-1)
        q, k, v = [tensor.unflatten(-1, (heads, width)) for tensor in packed.split(heads * width, dim=-1)]
        output = operation(q, k, v, g, beta, options["A_log"],
                           options["dt_bias"].view(q.shape[2], q.shape[-1]), None,
                           options["cp_context"])
        state = None
    else:
        output, state = operation(q=q, k=k, v=v, g=g, beta=beta, **options)
    assert state is None
    output.backward(do)
    assert all(leaf.grad is not None for leaf in leaves)
    return [output.detach(), *(leaf.grad.detach() for leaf in leaves)]


def worker(rank, size, rendezvous, root, case, output_dir):
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(minutes=55), device_id=torch.device("cuda", rank))
    try:
        target = upstream(Path(root))
        from tests.cp.utils import gather_global, shard_tensor
        from kernels.kda_module_dist import kcp
        from fla.utils import assert_close, get_abs_err, get_err_ratio

        assert kcp.chunk_kda is target.chunk_kda
        assert kcp.build_cp_context is target.build_cp_context
        assert inspect.signature(kcp.chunk_kda) == inspect.signature(target.chunk_kda)
        config = {key: value for key, value in case.items() if key not in ("id", "world_size", "lengths")}
        layout = config["layout"]
        lower_bound = -5.0 if config["safe_gate"] else None
        length = sum(case["lengths"])
        inputs = target._make_inputs(rank, length, 12, 128, torch.bfloat16,
                                     config["use_gate_in_kernel"], config["safe_gate"], lower_bound)
        q, k, v, g, beta, do, a_log, dt_bias = inputs
        offsets = [0, *torch.tensor(case["lengths"]).cumsum(0).tolist()]
        boundaries = torch.tensor(offsets, device="cuda", dtype=torch.int64)
        if rank == 0:
            offload = length > 16384
            print(f"KCP_REFERENCE_BEGIN {case['id']} saved_tensors_cpu={offload}", flush=True)
            # Preserve the exact upstream graph/operations/dtypes. Only saved
            # autograd tensor storage moves to CPU for the 64K reference.
            with torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext():
                reference = target._reference(q, k, v, g, beta, do, offsets, a_log, dt_bias,
                                              config["use_gate_in_kernel"], lower_bound)
            print(f"KCP_REFERENCE_END {case['id']}", flush=True)
        target._wait_for_reference(rank)
        dist.barrier()

        context_options = dict(cu_seqlens=boundaries, group=dist.group.WORLD, layout=layout)
        if config["use_tf32x3_affine_chain"] is not None:
            context_options["use_tf32x3_affine_chain"] = config["use_tf32x3_affine_chain"]
        options = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=config["use_gate_in_kernel"],
                       safe_gate=config["safe_gate"], lower_bound=lower_bound,
                       disable_recompute=config["disable_recompute"], state_v_first=config["state_v_first"],
                       A_log=a_log[:12].contiguous() if config["use_gate_in_kernel"] else None,
                       dt_bias=dt_bias if config["use_gate_in_kernel"] else None)
        local = [shard_tensor(tensor, rank, size, layout) for tensor in (q, k, v, g, beta)]
        do_local = shard_tensor(do, rank, size, layout).contiguous()
        native = call(target.chunk_kda, local, do_local,
                      dict(options, cp_context=target.build_cp_context(**context_options)))
        actual = call(kcp.chunk_kda, local, do_local,
                      dict(options, cp_context=kcp.build_cp_context(**context_options)))
        comparison = {name: errors(a, b) for name, a, b in zip(NAMES, actual, native)}
        adapter_comparison = None
        # The module adapter fixes these flags; compare it where they match.
        if config["use_gate_in_kernel"] and not config["disable_recompute"] and not config["state_v_first"]:
            logits = torch.randn_like(local[-1])
            adapter_inputs = [*local[:-1], logits]

            def native_adapter(q, k, v, g, beta, **kwargs):
                return target.chunk_kda(q, k, v, g, beta.sigmoid(), **kwargs)

            expected_adapter = call(native_adapter, adapter_inputs, do_local,
                                    dict(options, cp_context=target.build_cp_context(**context_options)))
            actual_adapter = call(kcp.module_recurrence, adapter_inputs, do_local,
                                  dict(options, cp_context=kcp.build_cp_context(**context_options)), adapter=True)
            adapter_comparison = {name: errors(a, b) for name, a, b in zip(NAMES, actual_adapter, expected_adapter)}
        passed = all(item["finite"] and item["exact"] for item in comparison.values())
        if adapter_comparison is not None:
            passed &= all(item["finite"] and item["exact"] for item in adapter_comparison.values())
        record = dict(case=case, rank=rank, device=torch.cuda.get_device_name(),
                      hostname=os.uname().nodename, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                      reference_saved_tensors_cpu=length > 16384,
                      shapes=[list(tensor.shape) for tensor in local],
                      dtypes=[str(tensor.dtype) for tensor in local],
                      local_boundaries=kcp.build_cp_context(**context_options).cu_seqlens.cpu().tolist(),
                      core_vs_upstream=comparison, module_adapter_vs_upstream=adapter_comparison,
                      pass_local=passed)
        # Gather only for the independent oracle, as in the original test.
        reference_checks = {}
        for name, tensor in zip(NAMES, actual):
            gathered = gather_global(tensor, size, layout)
            if rank == 0:
                expected = reference[NAMES.index(name)]
                item = errors(gathered, expected)
                # Record the upstream dtype-sensitive metric as well as FP64.
                item.update(upstream_ratio=get_err_ratio(expected, gathered),
                            upstream_max_abs=get_abs_err(expected, gathered),
                            warning_only=name in ("dg", "dbeta"), threshold=8e-3)
                item["pass"] = item["finite"] and (item["warning_only"] or
                                    item["upstream_max_abs"] <= 1e-6 or item["upstream_ratio"] < 8e-3)
                reference_checks[name] = item
                passed &= item["pass"]
                # Execute the upstream assertion policy, including diagnostics.
                try:
                    assert_close(name, expected, gathered, ratio=8e-3, warning=item["warning_only"])
                except AssertionError as error:
                    item.update({"pass": False, "assertion": str(error)})
                    passed = False
            del gathered
        if rank == 0:
            record["vs_recurrent_oracle"] = reference_checks
        flag = torch.tensor(int(passed), device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        record["pass"] = bool(flag.item())
        write(Path(output_dir) / case["id"] / f"rank{rank}.json", record)
        if not record["pass"]:
            raise AssertionError(record)
        if rank == 0:
            print(f"KCP_ALIGNMENT_CASE_PASS {case['id']} world_size={size}", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        dist.destroy_process_group()


def run(output, selected=None, resume=False, fla_root=DEFAULT_FLA, list_only=False):
    root = Path(fla_root).expanduser().resolve()
    target = upstream(root)
    grid = cases(target)
    if selected and selected != ["all"]:
        unknown = set(selected) - {case["id"] for case in grid}
        if unknown:
            raise ValueError(f"Unknown CP cases: {unknown}")
        grid = [case for case in grid if case["id"] in selected]
    if list_only:
        print(json.dumps(grid, indent=2))
        return
    output = Path(output).resolve()
    source_paths = [ROOT / "kcp.py", ROOT / "baseline.py", Path(__file__),
                    root / "tests/cp/test_kda.py", root / "tests/cp/utils.py"]
    identity = dict(fla_root=str(root),
                    fla_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    fla_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                    torch=torch.__version__, cuda_build=torch.version.cuda)
    if (output / "identity.json").exists():
        if not resume or json.loads((output / "identity.json").read_text()) != identity:
            raise ValueError("Existing accuracy evidence requires --resume and identical sources")
    else:
        write(output / "identity.json", identity)
    status = []
    for case in grid:
        size = case["world_size"]
        paths = [output / case["id"] / f"rank{rank}.json" for rank in range(size)]
        if not (resume and all(path.exists() and json.loads(path.read_text())["pass"] for path in paths)):
            if torch.cuda.device_count() < size:
                raise RuntimeError(f"{case['id']} requires {size} GPUs; cases must not be skipped")
            with TemporaryDirectory(prefix="kcp-alignment-", dir=output) as directory:
                rendezvous = (Path(directory) / "rendezvous").as_uri()
                mp.spawn(worker, args=(size, rendezvous, str(root), case, str(output)), nprocs=size, join=True)
        status.append(dict(id=case["id"], world_size=size, **{"pass": all(json.loads(path.read_text())["pass"] for path in paths)}))
        write(output / "summary.json", dict(cases=status, requested_cases=len(grid),
                                            complete=len(status) == len(grid),
                                            all_pass=all(item["pass"] for item in status)))
    print(f"KCP_ALIGNMENT_COMPLETE cases={len(status)} all_pass=True output={output}", flush=True)
