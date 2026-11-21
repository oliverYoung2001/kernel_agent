"""Lazily construct all explicit arguments from twenty input recipes."""

from __future__ import annotations

import json
from itertools import accumulate
from pathlib import Path

import torch

CASE_DIRECTORY = Path(__file__).resolve().parent
MANIFEST_PATH = CASE_DIRECTORY / "cases.json"
ARCHIVED_MANIFEST_PATH = CASE_DIRECTORY / "cases_0.json"
INPUTS_PATH = Path(__file__).resolve()

def _read_cases(path):
    manifest = json.loads(Path(path).read_text())
    common = {**manifest["defaults"], "full_module": manifest["schema_version"] >= 3}
    return tuple({**common, **case} for case in manifest["cases"])


CASES = _read_cases(MANIFEST_PATH)
if len(CASES) != 20 or len({case["id"] for case in CASES}) != 20:
    raise ValueError("The KDA case manifest must contain exactly twenty unique cases")


def _validate_recipe(case: dict) -> None:
    """Validate real sequence boundaries even when tensors will be on meta."""
    dimensions = ("batch", "sequence", "hidden", "hqk", "hv", "dk", "dv", "conv_width")
    if any(type(case[name]) is not int or case[name] < 1 for name in dimensions):
        raise ValueError(f"{case['id']}: dimensions must be positive integers")
    if case["hv"] % case["hqk"]:
        raise ValueError(f"{case['id']}: Hv must be divisible by Hqk")
    if case.get("variable_length", False):
        lengths = case.get("sequence_lengths", [])
        if len(lengths) != case["batch"] or any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError(f"{case['id']}: one positive sequence length is required per sample")
        total = sum(lengths)
        if total != case["sequence"]:
            raise ValueError(f"{case['id']}: packed sequence must equal the sum of sample lengths")
    else:
        if "sequence_lengths" in case:
            raise ValueError(f"{case['id']}: sequence_lengths requires variable_length=true")
        total = case["batch"] * case["sequence"]
    if case.get("total_tokens", total) != total:
        raise ValueError(f"{case['id']}: inconsistent total_tokens")


for _case in CASES:
    _validate_recipe(_case)


def load_cases(manifest_path: str | Path = MANIFEST_PATH) -> tuple[dict, ...]:
    """Load an explicit manifest without changing the active defaults."""
    cases = _read_cases(manifest_path)
    if len(cases) != 20 or len({case["id"] for case in cases}) != 20:
        raise ValueError("A KDA manifest must contain twenty unique cases")
    for case in cases:
        _validate_recipe(case)
    return cases


def get_case(case_id: str, *, manifest_path: str | Path | None = None) -> dict:
    """Return a copy so callers cannot change the manifest through this API."""
    for case in CASES if manifest_path is None else load_cases(manifest_path):
        if case["id"] == case_id:
            return {key: list(value) if isinstance(value, list) else value for key, value in case.items()}
    raise KeyError(f"Unknown KDA case: {case_id}")


def make_inputs(case_id: str, *, device: str | torch.device = "cuda", manifest_path: str | Path | None = None) -> dict:
    """Build reference(**arguments) for one of the twenty manifest entries."""
    return make_inputs_from_recipe(get_case(case_id, manifest_path=manifest_path), device=device)


def make_inputs_from_recipe(case: dict, *, device: str | torch.device = "cuda") -> dict:
    """Materialize the current module's weights and caller-owned buffers."""
    if not case["full_module"]:
        raise ValueError("Historical input generation is archived in experiment0/reports/history/before-single-call.zip")
    _validate_recipe(case)
    device = torch.device(device)
    generator = torch.Generator(device="cpu" if device.type == "meta" else device)
    generator.manual_seed(case["seed"])
    act = getattr(torch, case["dtype"])
    n, s, m = case["batch"], case["sequence"], case["hidden"]
    b = 1 if case["variable_length"] else n
    hq, hv, dk, dv = case["hqk"], case["hv"], case["dk"], case["dv"]
    c = 2 * hq * dk + hv * dv

    def randn(shape, *, dtype=act, std=1.0):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator).mul_(std)

    return {
        "inpx": randn((b, s, m)),
        "out": torch.full((b, s, m), float("nan"), device=device, dtype=act),
        "final_state": torch.full((n, hv, dk, dv), float("nan"), device=device, dtype=torch.float32),
        "input_norm_weight": torch.ones(m, device=device, dtype=act),
        "in_proj_qkvbfg_a_weight": randn((m, c + hv + dk + dv), std=m ** -0.5),
        "conv1d_weight": randn((c, 1, case["conv_width"]), std=0.2),
        "f_b_proj_weight": randn((dk, hv * dk), std=dk ** -0.5),
        "g_b_proj_weight": randn((dv, hv * dv), std=dv ** -0.5),
        "a_log_weight": torch.zeros(hv, device=device, dtype=torch.float32),
        "dt_bias_weight": torch.zeros((hv, dk), device=device, dtype=torch.float32),
        "norm_weight": torch.ones(dv, device=device, dtype=act),
        "o_proj_weight": randn((hv * dv, m), std=(hv * dv) ** -0.5),
    }


def run_case(operation, arguments, case):
    """Pass the whole batch or one jagged view of all packed contexts."""
    if not case.get("variable_length", False):
        return operation(**arguments)
    x, y = arguments["inpx"], arguments["out"]
    offsets = torch.tensor(list(accumulate(case["sequence_lengths"], initial=0)),
                           dtype=torch.int64, device=x.device)
    packed = dict(arguments)
    packed["inpx"] = torch.nested.nested_tensor_from_jagged(x.squeeze(0), offsets)
    packed["out"] = torch.nested.nested_tensor_from_jagged(y.squeeze(0), offsets)
    return operation(**packed)
