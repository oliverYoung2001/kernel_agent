"""Small distributed checks; these do not qualify the production manifests."""
from __future__ import annotations

import argparse
import inspect
import json
import os

import torch
import torch.distributed as dist

from ..baseline import baseline_kcp, baseline_tp, baseline_ulysses
from ..cases.generate_cases import LENGTHS
from ..cases.inputs import CASES, make_inputs_from_recipe
from ..distributed import (
    WEIGHT_NAMES, gather_tokens, head_to_sequence, partition_weights, scatter_sum,
    sequence_to_head, topology,
)
from ..reference import reference
from ..run import metrics


def check_manifest():
    assert len(CASES) == 20
    for model in {case["model"] for case in CASES}:
        cases = [case for case in CASES if case["model"] == model]
        assert sorted(case["parallel_size"] for case in cases) == [8, 8, 16, 16, 16]
        for p in (8, 16):
            packed = [case for case in cases if case["parallel_size"] == p and case["variable_length"]]
            assert len(packed) == 1 and packed[0]["sequence_lengths"] == LENGTHS
        fixed = sorted((case["parallel_size"], case["sequence"]) for case in cases if not case["variable_length"])
        assert fixed == [(8, 131072), (16, 131072), (16, 262144)]


def recipe(p, kind):
    total = p * 128
    case = dict(id=f"small_{kind}_p{p}", seed=173, hidden=64, hqk=2 * p, hv=4 * p,
                dk=128, dv=128, conv_width=4, parallel_size=p, dtype="bfloat16",
                variable_length=kind != "dense", batch=1, sequence=total, total_tokens=total)
    if kind != "dense":
        case.update(batch=3, sequence_lengths=[total // 3 + 1, 17, total - total // 3 - 18])
    return case


def check_collectives(device):
    p, rank = topology()
    # The expected result is indexed independently from the collective reshapes.
    global_tensor = torch.arange(p * 7 * p * 3 * 5, device=device).reshape(1, p * 7, p * 3, 5)
    local = global_tensor[:, rank * 7:(rank + 1) * 7].contiguous()
    heads = sequence_to_head(local)
    assert torch.equal(heads, global_tensor[:, :, rank * 3:(rank + 1) * 3])
    assert torch.equal(head_to_sequence(heads), local)
    assert torch.equal(gather_tokens(local.squeeze(0)), global_tensor.squeeze(0))
    beta = local[..., 0]
    assert torch.equal(head_to_sequence(sequence_to_head(beta)), beta)
    contribution = torch.arange(p * 7 * 4, device=device, dtype=torch.float32).reshape(p * 7, 4) + rank
    output = torch.empty((7, 4), device=device)
    scatter_sum(contribution, output)
    expected = p * (contribution - rank) + p * (p - 1) / 2
    assert torch.equal(output, expected[rank * 7:(rank + 1) * 7])


def check_generator():
    p, rank = topology()
    case = recipe(p, "packed")
    before = torch.random.get_rng_state().clone()
    full = make_inputs_from_recipe(case, implementation="reference", device="cpu", rank=rank)
    tp = make_inputs_from_recipe(case, implementation="tp", device="cpu", rank=rank)
    automatic = make_inputs_from_recipe(case, implementation="tp", device="cpu")
    assert torch.equal(before, torch.random.get_rng_state())
    shard = partition_weights({name: full[name] for name in WEIGHT_NAMES},
                              hq=case["hqk"], hv=case["hv"], dk=128, dv=128, rank=rank, world_size=p)
    assert torch.equal(full["inpx"], tp["inpx"])
    assert torch.equal(automatic["inpx"], tp["inpx"])
    assert all(torch.equal(tp[name], shard[name]) for name in WEIGHT_NAMES)
    assert all(torch.equal(automatic[name], tp[name]) for name in WEIGHT_NAMES)
    assert not torch.isfinite(tp["out"]).any()
    for operation in (reference, baseline_tp, baseline_ulysses, baseline_kcp):
        assert "final_state" not in inspect.signature(operation).parameters
        inspect.signature(operation).bind(**tp)


def check_gpu():
    p, rank = topology()
    checks = []
    for kind in ("dense", "packed", "weak_decay", "residual"):
        case = recipe(p, kind)
        truth = make_inputs_from_recipe(case, implementation="reference", rank=rank)
        if kind == "weak_decay":
            truth["dt_bias_weight"].fill_(-8)
        if kind == "residual":
            truth["o_proj_weight"].zero_()
        reference(**truth)
        expected = truth["out"].clone()
        if kind == "residual":
            assert torch.equal(expected, truth["inpx"])
        del truth
        for name, operation in (("tp", baseline_tp), ("ulysses", baseline_ulysses), ("kcp", baseline_kcp)):
            arguments = make_inputs_from_recipe(case, implementation=name, rank=rank)
            if kind == "weak_decay":
                arguments["dt_bias_weight"].fill_(-8)
            if kind == "residual":
                arguments["o_proj_weight"].zero_()
            inputs = {key: value.clone() for key, value in arguments.items() if isinstance(value, torch.Tensor) and key != "out"}
            operation(**arguments)
            result = metrics(arguments["out"], expected)
            assert result["finite"] and result["scaled_max"] <= 0.01 and result["rmse_ratio"] <= 0.01, (kind, name, result)
            assert all(torch.equal(arguments[key], value) for key, value in inputs.items())
            if kind == "residual":
                assert torch.equal(arguments["out"], arguments["inpx"])
            checks.append(dict(case=case["id"], implementation=name, **result))
            if rank == 0:
                print(json.dumps(checks[-1]), flush=True)
            del arguments, inputs
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if args.gpu else torch.device("cpu")
    if args.gpu:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl" if args.gpu else "gloo")
    try:
        check_manifest()
        check_collectives(device)
        check_generator()
        if args.gpu:
            check_gpu()
        if dist.get_rank() == 0:
            print(f"KDA_MODULE_DIST_CHECKS_COMPLETE world_size={dist.get_world_size()} gpu={args.gpu}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
