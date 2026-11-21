"""Per-output numerical gates for KDA."""
import torch

from ..cases.inputs import run_case
from ..runner import PROTOCOL


def copy_args(a):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in a.items()}


def compare(a, expected):
    metrics = {}
    for name in ('out', 'final_state'):
        actual, ref = a[name], expected[name]
        if actual.shape != ref.shape or actual.dtype != ref.dtype:
            raise ValueError(f'{name}: output shape or dtype differs from ground truth')
        av, rv = actual.flatten(), ref.flatten()
        max_error, max_ref, error_energy, ref_energy = 0., 0., 0., 0.
        for start in range(0, av.numel(), 1048576):
            x = av[start:start + 1048576].double()
            r = rv[start:start + 1048576].to(device=x.device, dtype=torch.float64)
            if not bool(torch.isfinite(x).all() & torch.isfinite(r).all()):
                raise ValueError(f'{name}: nonfinite')
            d = x - r
            max_error = max(max_error, d.abs().max().item())
            max_ref = max(max_ref, r.abs().max().item())
            error_energy += d.square().sum().item()
            ref_energy += r.square().sum().item()
        sm = max_error / max(max_ref, PROTOCOL['epsilon_scale'])
        rmse = (error_energy / max(ref_energy, PROTOCOL['epsilon_energy'])) ** .5
        rules = PROTOCOL['thresholds'][name]
        metrics[name] = {'scaled_max': sm, 'rmse_ratio': rmse,
                         'pass': sm <= rules['scaled_max'] and rmse <= rules['rmse_ratio']}
    return all(x['pass'] for x in metrics.values()), metrics


def verify_candidate(fn, arguments, case, expected):
    """Run a candidate to completion and compare both caller-owned outputs."""
    run_case(fn, arguments, case)
    torch.cuda.synchronize()
    return compare(arguments, expected)
