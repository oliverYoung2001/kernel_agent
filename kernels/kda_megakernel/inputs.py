"""Lazily construct all explicit arguments from twenty input recipes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

_manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
CASES = tuple({**_manifest["defaults"], **case} for case in _manifest["cases"])
if len(CASES) != 20 or len({case["id"] for case in CASES}) != 20:
    raise ValueError("The KDA case manifest must contain exactly twenty unique cases")


def get_case(case_id: str) -> dict:
    """Return a copy so callers cannot change the manifest through this API."""
    for case in CASES:
        if case["id"] == case_id:
            return dict(case)
    raise KeyError(f"Unknown KDA case: {case_id}")


def make_inputs(case_id: str, *, device: str | torch.device = "cpu") -> dict:
    """Build reference(**arguments), without changing the global RNG state.

    Use device='meta' to validate the large case without allocating its data.
    Numerical generation is reproducible on the same device/backend/version;
    CPU and CUDA random generators need not produce identical values.
    """
    case = get_case(case_id)
    device = torch.device(device)
    rng_device = "cpu" if device.type == "meta" else device
    generator = torch.Generator(device=rng_device).manual_seed(case["seed"])
    act = getattr(torch, case["dtype"])
    b, s, m = case["batch"], case["sequence"], case["hidden"]
    hq, hv, dk, dv = case["hqk"], case["hv"], case["dk"], case["dv"]
    c = 2 * hq * dk + hv * dv

    def normal(shape):
        return torch.randn(shape, device=device, generator=generator, dtype=torch.float32)

    def uniform(shape):
        return torch.rand(shape, device=device, generator=generator, dtype=torch.float32)

    def activation(shape):
        mode = case["distribution"]
        if mode == "normal":
            value = normal(shape)
        elif mode == "uniform":
            value = uniform(shape) * 2 - 1
        elif mode == "rademacher":
            value = (uniform(shape) >= 0.5).float() * 2 - 1
        elif mode == "zeros":
            value = torch.zeros(shape, device=device)
        elif mode == "sparse":
            value = normal(shape) * (uniform(shape) < 0.05)
        elif mode == "spiky":
            value = normal(shape) * torch.where(uniform(shape) < 0.01, 32.0, 1.0)
        else:
            raise ValueError(f"Unknown distribution: {mode}")
        return (value * case["input_scale"]).to(act)

    projection_std = case["projection_std"]
    out_projection_std = case["out_projection_std"]
    projection_std = 1 / math.sqrt(m) if projection_std is None else projection_std
    out_projection_std = 1 / math.sqrt(hv * dv) if out_projection_std is None else out_projection_std
    arguments = {
        "inpx": activation((b, s, m)),
        "in_proj_qkv_w": (normal((m, c)) * projection_std).to(act),
        "conv1d_weight": (normal((c, 1, case["conv_width"])) * case["conv_std"]).to(act),
    }
    alpha = -uniform((b, s, hv, dk)) * case["decay_scale"]
    if case["decay_mode"] == "none":
        alpha.zero_()
    elif case["decay_mode"] == "mixed":
        multiplier = torch.where(torch.arange(s, device=device) % 2 == 0, 0.001, 3.0)
        alpha = alpha * multiplier.reshape(1, s, 1, 1)
    elif case["decay_mode"] != "uniform":
        raise ValueError("Unknown decay mode")
    beta = uniform((b, s, hv))
    if case["beta_mode"] == "zeros":
        beta.zero_()
    elif case["beta_mode"] == "ones":
        beta.fill_(1)
    elif case["beta_mode"] == "binary":
        beta = (beta >= 0.5).float()
    elif case["beta_mode"] != "uniform":
        raise ValueError("Unknown beta mode")
    arguments.update({
        "alpha_log": alpha,
        "beta": beta.to(act),
        "gate": (normal((b, s, hv, dv)) * case["gate_scale"]).to(act),
        "norm_weight": torch.ones((dv,), device=device, dtype=act),
        "o_proj": (normal((hv * dv, m)) * out_projection_std).to(act),
        "initial_state": normal((b, hv, dk, dv)) * case["state_std"],
        "residual": (normal((b, s, m)) * case["residual_std"]).to(act),
        "out": torch.full((b, s, m), float("nan"), device=device, dtype=act),
        "final_state": torch.full((b, hv, dk, dv), float("nan"), device=device, dtype=torch.float32),
        "num_heads_qk": hq,
        "num_heads_v": hv,
        "head_dim_qk": dk,
        "head_dim_v": dv,
        "l2_eps": case["l2_eps"],
        "norm_eps": case["norm_eps"],
        "scale": dk ** -0.5,
    })
    return arguments
