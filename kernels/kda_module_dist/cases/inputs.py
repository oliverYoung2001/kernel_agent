"""Deterministic rank-local inputs, with directly generated TP weight shards."""
from __future__ import annotations

from itertools import accumulate
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from ..distributed import topology, weight_indices, weight_shapes

MANIFEST_PATH = Path(__file__).with_name("cases.json")


def validate_recipe(case):
    for name in ("batch", "sequence", "hidden", "hqk", "hv", "dk", "dv", "conv_width", "parallel_size"):
        if type(case[name]) is not int or case[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if case["hv"] % case["hqk"]:
        raise ValueError("Hv must be divisible by Hq")
    if case["dtype"] not in ("bfloat16", "float16", "float32"):
        raise ValueError("Unsupported activation dtype")
    if case.get("variable_length", False):
        lengths = case.get("sequence_lengths", [])
        if len(lengths) != case["batch"] or any(type(n) is not int or n < 1 for n in lengths) or sum(lengths) != case["sequence"]:
            raise ValueError("Packed lengths must be positive and sum to sequence")
        total = sum(lengths)
    else:
        if "sequence_lengths" in case:
            raise ValueError("sequence_lengths requires variable_length=true")
        total = case["batch"] * case["sequence"]
    if case.get("total_tokens", total) != total or total % case["parallel_size"]:
        raise ValueError("total_tokens must be correct and divisible by parallel_size")
    if type(case["seed"]) is not int:
        raise ValueError("seed must be an integer")


def load_cases(manifest_path=MANIFEST_PATH):
    manifest = json.loads(Path(manifest_path).read_text())
    cases = tuple({**manifest["defaults"], **case} for case in manifest["cases"])
    if len(cases) != 20 or len({case["id"] for case in cases}) != 20:
        raise ValueError("Expected twenty unique distributed cases")
    for case in cases:
        validate_recipe(case)
    return cases


CASES = load_cases()


def get_case(case_id, *, manifest_path=MANIFEST_PATH):
    for case in load_cases(manifest_path):
        if case["id"] == case_id:
            return json.loads(json.dumps(case))
    raise KeyError(case_id)


def _normal(shape, *, seed, stream, dtype, device, std=1.0, selection=None):
    """CPU row-block seeds permit exact slices without a full tensor allocation."""
    rows, columns = shape[0], math.prod(shape[1:])
    row_ids, column_ids = torch.arange(rows), torch.arange(columns)
    if selection is not None:
        axis, indices = selection
        if axis == 0:
            row_ids = torch.tensor(indices, dtype=torch.long)
        elif axis == 1 and len(shape) == 2:
            column_ids = torch.tensor(indices, dtype=torch.long)
        else:
            raise ValueError("Only leading rows or matrix columns can be sharded")
    output_shape = list(shape)
    if selection is not None:
        output_shape[selection[0]] = len(selection[1])
    if device.type == "meta":
        return torch.empty(output_shape, device=device, dtype=dtype)
    result = torch.empty((row_ids.numel(), column_ids.numel()), dtype=dtype)
    generator = torch.Generator(device="cpu")
    for block in torch.unique(row_ids // 128).tolist():
        begin = block * 128
        generator.manual_seed(seed + 1000003 * stream + block)
        value = torch.randn((min(128, rows - begin), columns), generator=generator, dtype=dtype).mul_(std)
        selected = (row_ids >= begin) & (row_ids < begin + 128)
        result[selected] = value.index_select(0, row_ids[selected] - begin).index_select(1, column_ids)
    return result.reshape(output_shape).to(device=device).contiguous()


def make_inputs(case_id, *, implementation="tp", device="cuda", rank=None, manifest_path=MANIFEST_PATH, group=None):
    return make_inputs_from_recipe(get_case(case_id, manifest_path=manifest_path),
                                   implementation=implementation, device=device, rank=rank, group=group)


def make_inputs_from_recipe(case, *, implementation="tp", device="cuda", rank=None, group=None):
    validate_recipe(case)
    if implementation not in ("reference", "tp", "ulysses", "kcp"):
        raise ValueError(f"Unknown implementation: {implementation}")
    p = case["parallel_size"]
    device = torch.device(device)
    if rank is None:
        rank = topology(group)[1]
    if dist.is_initialized() and device.type != "meta" and topology(group)[0] != p:
        raise ValueError("The process-group size must equal the recipe's parallel_size")
    if type(rank) is not int or not 0 <= rank < p:
        raise ValueError("rank must lie inside the case's parallel group")
    dtype = getattr(torch, case["dtype"])
    hq, hv, dk, dv, hidden = (case[k] for k in ("hqk", "hv", "dk", "dv", "hidden"))
    if implementation == "ulysses" and (hq % p or hv % p):
        raise ValueError("Ulysses requires head counts divisible by parallel_size")
    selections = weight_indices(hq, hv, dk, dv, rank, p) if implementation == "tp" else {}
    lengths = case["sequence_lengths"] if case.get("variable_length", False) else [case["sequence"]] * case["batch"]
    total = sum(lengths)
    local = total // p
    x = _normal((total, hidden), seed=case["seed"], stream=0, dtype=dtype, device=device,
                selection=(0, list(range(rank * local, (rank + 1) * local))))
    result = dict(inpx=x, out=torch.full_like(x, float("nan")),
                  cu_seqlens=torch.tensor(list(accumulate(lengths, initial=0)), dtype=torch.int64, device=device),
                  hq=hq, hv=hv, dk=dk, dv=dv, group=group)
    shapes = weight_shapes(hidden, hq, hv, dk, dv, case["conv_width"])
    scales = dict(in_proj_qkvbfg_a_weight=hidden ** -0.5, conv1d_weight=0.2,
                  f_b_proj_weight=dk ** -0.5, g_b_proj_weight=dv ** -0.5,
                  o_proj_weight=(hv * dv) ** -0.5)
    for stream, (name, shape) in enumerate(shapes.items(), start=1):
        selection = selections.get(name)
        weight_dtype = torch.float32 if name in ("a_log_weight", "dt_bias_weight") else dtype
        if name in ("input_norm_weight", "norm_weight", "a_log_weight", "dt_bias_weight"):
            local_shape = list(shape)
            if selection is not None:
                local_shape[selection[0]] = len(selection[1])
            value = 0.0 if weight_dtype == torch.float32 and name in ("a_log_weight", "dt_bias_weight") else 1.0
            result[name] = torch.full(local_shape, value, device=device, dtype=weight_dtype)
        else:
            result[name] = _normal(shape, seed=case["seed"], stream=stream, dtype=weight_dtype,
                                   device=device, std=scales[name], selection=selection)
    return result


def run_case(operation, arguments, case=None):
    return operation(**arguments)
