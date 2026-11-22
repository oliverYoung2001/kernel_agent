"""Distributed full-module oracle, exposing only the caller-owned out buffer."""
from __future__ import annotations

import torch

from .distributed import gather_tokens, topology, validate_input_specs


@torch.no_grad()
def reference(
    inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
    f_b_proj_weight, g_b_proj_weight, a_log_weight, dt_bias_weight, norm_weight,
    o_proj_weight, cu_seqlens, hq, hv, dk, dv, group=None,
):
    """Gather tokens, apply the FP32 token oracle, and write the local slice."""
    from kernels.kda_megakernel.reference import reference as full_reference

    weights = dict(input_norm_weight=input_norm_weight, in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight,
                   conv1d_weight=conv1d_weight, f_b_proj_weight=f_b_proj_weight,
                   g_b_proj_weight=g_b_proj_weight, a_log_weight=a_log_weight,
                   dt_bias_weight=dt_bias_weight, norm_weight=norm_weight, o_proj_weight=o_proj_weight)
    p, rank = topology(group)
    validate_input_specs(inpx, out, cu_seqlens=cu_seqlens, hq=hq, hv=hv, dk=dk, dv=dv,
                         weights=weights, implementation="reference", world_size=p)
    x = gather_tokens(inpx, group)
    y = torch.empty_like(x)
    packed_x = torch.nested.nested_tensor_from_jagged(x, cu_seqlens)
    packed_y = torch.nested.nested_tensor_from_jagged(y, cu_seqlens)
    # The legacy oracle stores its internal recurrence state in scratch storage.
    scratch = torch.empty((cu_seqlens.numel() - 1, hv, dk, dv), device=x.device, dtype=torch.float32)
    full_reference(packed_x, packed_y, scratch, **weights)
    out.copy_(y.narrow(0, rank * inpx.shape[0], inpx.shape[0]))
