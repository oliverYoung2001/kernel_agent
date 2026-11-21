"""Native FLA and FlashKDA attention baselines with a single backend call."""
from __future__ import annotations

import inspect

import torch

from .reference import LOWER_BOUND, _attention, normalize_qk


def _fla_recurrence(q, k, v, decay, beta, a_log, dt_bias, final, offsets):
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import kda_gate_fwd

    q, k = normalize_qk(q), normalize_qk(k)
    alpha_log = kda_gate_fwd(decay, a_log, dt_bias, lower_bound=LOWER_BOUND)
    attention, end = inspect.unwrap(chunk_kda)(
        q=q, k=k, v=v.contiguous(), g=alpha_log, beta=beta.float().sigmoid(),
        scale=q.shape[-1] ** -0.5, output_final_state=True,
        safe_gate=True, cu_seqlens=offsets,
    )
    final.copy_(end)
    return attention


def _flash_recurrence(q, k, v, decay, beta, a_log, dt_bias, final, offsets):
    import flash_kda

    if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("FlashKDA requires BF16 and Dk=Dv=128")
    group = v.shape[2] // q.shape[2]
    if group != 1:
        q, k = q.repeat_interleave(group, dim=2), k.repeat_interleave(group, dim=2)
    attention = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    end = torch.empty_like(final)
    flash_kda.fwd(
        q.contiguous(), k.contiguous(), v.contiguous(), decay, beta, 128 ** -0.5,
        attention, a_log, dt_bias, LOWER_BOUND, final_state=end, cu_seqlens=offsets,
    )
    final.copy_(end.transpose(-1, -2))
    return attention


@torch.no_grad()
def baseline_fla(
    inpx, out, final_state, *, input_norm_weight, in_proj_qkvbfg_a_weight,
    conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
    dt_bias_weight, norm_weight, o_proj_weight,
):
    """Run FLA once for the full dense batch or all packed contexts."""
    _attention(
        _fla_recurrence, inpx, out, final_state, input_norm_weight=input_norm_weight,
        in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight, conv1d_weight=conv1d_weight,
        f_b_proj_weight=f_b_proj_weight, g_b_proj_weight=g_b_proj_weight,
        a_log_weight=a_log_weight, dt_bias_weight=dt_bias_weight,
        norm_weight=norm_weight, o_proj_weight=o_proj_weight,
    )


@torch.no_grad()
def baseline_flashkda(
    inpx, out, final_state, *, input_norm_weight, in_proj_qkvbfg_a_weight,
    conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
    dt_bias_weight, norm_weight, o_proj_weight,
):
    """Run FlashKDA once for the full dense batch or all packed contexts."""
    _attention(
        _flash_recurrence, inpx, out, final_state, input_norm_weight=input_norm_weight,
        in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight, conv1d_weight=conv1d_weight,
        f_b_proj_weight=f_b_proj_weight, g_b_proj_weight=g_b_proj_weight,
        a_log_weight=a_log_weight, dt_bias_weight=dt_bias_weight,
        norm_weight=norm_weight, o_proj_weight=o_proj_weight,
    )
