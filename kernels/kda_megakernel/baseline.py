"""PyTorch + FLA/FlashKDA + PyTorch baselines with the reference interface.

Adapted from LACP/lacp/la_ops/kda/three_way.py and the installed backends.
Optional GPU packages are imported only when their baseline is called.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from .reference import validate_arguments

__all__ = ["baseline_fla", "baseline_flashkda"]


def _prepare_qkv(inpx, in_proj_qkv_w, conv1d_weight, hq, hv, dk, dv):
    """Project and convolve in PyTorch, keeping SiLU before the act cast."""
    b, s, _ = inpx.shape
    projected = torch.matmul(inpx, in_proj_qkv_w)
    convolved = F.conv1d(
        projected.transpose(1, 2).float(), conv1d_weight.float(),
        padding=conv1d_weight.shape[-1] - 1, groups=conv1d_weight.shape[0],
    )[:, :, :s]
    packed = F.silu(convolved).transpose(1, 2).to(inpx.dtype)
    q, k, v = packed.split((hq * dk, hq * dk, hv * dv), dim=-1)
    return q.reshape(b, s, hq, dk), k.reshape(b, s, hq, dk), v.reshape(b, s, hv, dv)


def _normalize(x, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _pad_last(x, width):
    if x.shape[-1] != width:
        x = F.pad(x, (0, width - x.shape[-1]))
    return x.contiguous()


def _pad_state(state, dk, dv):
    if state.shape[-2:] == (dk, dv):
        return state.clone()
    return F.pad(state, (0, dv - state.shape[-1], 0, dk - state.shape[-2]))


def _finish(attention, gate, norm_weight, o_proj, residual, out, norm_eps):
    """Apply PyTorch gated RMSNorm, output projection, and residual addition."""
    attention = attention.to(out.dtype).float()
    normalized = attention * torch.rsqrt(attention.square().mean(-1, keepdim=True) + norm_eps)
    normalized = (normalized * norm_weight.float() * gate.float().sigmoid()).to(out.dtype)
    b, s, m = out.shape
    torch.addmm(
        residual.reshape(b * s, m), normalized.reshape(b * s, -1), o_proj,
        out=out.view(b * s, m),
    )


def _flash_logits(alpha_log, beta):
    """Invert activated gates without clamping or changing caller inputs.

    FlashKDA uses alpha=-5*sigmoid(g) and beta=sigmoid(b). Endpoints
    intentionally become infinite logits. BF16 logit rounding remains part
    of this baseline's numerical error; it is not an exact round trip.
    """
    if not bool((torch.isfinite(alpha_log) & (alpha_log >= -5) & (alpha_log <= 0)).all()):
        raise ValueError("FlashKDA requires finite alpha_log in [-5, 0]")
    if not bool((torch.isfinite(beta) & (beta >= 0) & (beta <= 1)).all()):
        raise ValueError("FlashKDA requires finite beta in [0, 1]")
    return torch.logit(alpha_log / -5.0).to(torch.bfloat16).contiguous(), \
        torch.logit(beta.float()).to(torch.bfloat16).contiguous()


@torch.no_grad()
def baseline_fla(
    inpx: torch.Tensor,
    in_proj_qkv_w: torch.Tensor,
    conv1d_weight: torch.Tensor,
    alpha_log: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    o_proj: torch.Tensor,
    initial_state: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    *,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim_qk: int,
    head_dim_v: int,
    l2_eps: float = 1e-6,
    norm_eps: float = 1e-5,
    scale: float | None = None,
) -> None:
    """Run PyTorch preprocessing, native FLA chunk_kda, and PyTorch postprocessing."""
    validate_arguments(
        inpx, in_proj_qkv_w, conv1d_weight, alpha_log, beta, gate,
        norm_weight, o_proj, initial_state, residual, out, final_state,
        num_heads_qk=num_heads_qk, num_heads_v=num_heads_v,
        head_dim_qk=head_dim_qk, head_dim_v=head_dim_v,
        l2_eps=l2_eps, norm_eps=norm_eps, scale=scale,
    )
    if inpx.device.type != "cuda":
        raise ValueError("baseline_fla requires CUDA tensors")
    if head_dim_qk > 256:
        raise ValueError("The installed FLA chunk_kda supports head_dim_qk <= 256")
    from fla.ops.kda import chunk_kda

    q, k, v = _prepare_qkv(
        inpx, in_proj_qkv_w, conv1d_weight,
        num_heads_qk, num_heads_v, head_dim_qk, head_dim_v,
    )
    q, k = _normalize(q, l2_eps), _normalize(k, l2_eps)
    # Small head dimensions need at least a Tensor Core operand tile.
    # Zero padding adds no dot work to the mathematical problem.
    dk, dv = max(32, head_dim_qk), max(32, head_dim_v)
    q, k, v = _pad_last(q, dk), _pad_last(k, dk), _pad_last(v, dv)
    g = _pad_last(alpha_log, dk)
    state = _pad_state(initial_state, dk, dv)
    # Bypass optional third-party dispatch so this baseline is always FLA.
    native_chunk_kda = inspect.unwrap(chunk_kda)
    attention, state = native_chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=head_dim_qk ** -0.5 if scale is None else scale,
        initial_state=state, output_final_state=True,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False, safe_gate=True,
        transpose_state_layout=False,
    )
    _finish(attention[..., :head_dim_v], gate, norm_weight, o_proj, residual, out, norm_eps)
    final_state.copy_(state[..., :head_dim_qk, :head_dim_v])


@torch.no_grad()
def baseline_flashkda(
    inpx: torch.Tensor,
    in_proj_qkv_w: torch.Tensor,
    conv1d_weight: torch.Tensor,
    alpha_log: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    o_proj: torch.Tensor,
    initial_state: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    final_state: torch.Tensor,
    *,
    num_heads_qk: int,
    num_heads_v: int,
    head_dim_qk: int,
    head_dim_v: int,
    l2_eps: float = 1e-6,
    norm_eps: float = 1e-5,
    scale: float | None = None,
) -> None:
    """Run PyTorch preprocessing, unchanged flash_kda.fwd, and PyTorch postprocessing.

    Requires BF16 CUDA inputs and Dk,Dv <=128. Q/K head replication and
    zero padding adapt GVA/smaller dimensions to the backend's 128x128 API.
    The backend fixes l2_eps=1e-6; another value is rejected explicitly.
    """
    validate_arguments(
        inpx, in_proj_qkv_w, conv1d_weight, alpha_log, beta, gate,
        norm_weight, o_proj, initial_state, residual, out, final_state,
        num_heads_qk=num_heads_qk, num_heads_v=num_heads_v,
        head_dim_qk=head_dim_qk, head_dim_v=head_dim_v,
        l2_eps=l2_eps, norm_eps=norm_eps, scale=scale,
    )
    if inpx.device.type != "cuda":
        raise ValueError("baseline_flashkda requires CUDA tensors")
    if inpx.dtype != torch.bfloat16:
        raise ValueError("The installed FlashKDA backend requires bfloat16 activations")
    if max(head_dim_qk, head_dim_v) > 128:
        raise ValueError("FlashKDA requires head_dim_qk and head_dim_v <= 128 with padding")
    if l2_eps != 1e-6:
        raise ValueError("The installed FlashKDA backend fixes l2_eps=1e-6")
    import flash_kda

    raw_g, raw_beta = _flash_logits(alpha_log, beta)
    q, k, v = _prepare_qkv(
        inpx, in_proj_qkv_w, conv1d_weight,
        num_heads_qk, num_heads_v, head_dim_qk, head_dim_v,
    )
    # FlashKDA performs its own Q/K L2 normalization with epsilon 1e-6.
    group = num_heads_v // num_heads_qk
    if group != 1:
        q, k = q.repeat_interleave(group, dim=2), k.repeat_interleave(group, dim=2)
    q, k, v = _pad_last(q, 128), _pad_last(k, 128), _pad_last(v, 128)
    # Padded gate channels have zero keys/state, so their decay is irrelevant.
    raw_g = F.pad(raw_g, (0, 128 - head_dim_qk), value=-float("inf"))
    state_in = _pad_state(initial_state, 128, 128).transpose(-1, -2).contiguous()
    state_out = torch.empty_like(state_in)
    attention = torch.empty_like(v)
    a_log = torch.zeros(num_heads_v, device=inpx.device, dtype=torch.float32)
    dt_bias = torch.zeros((num_heads_v, 128), device=inpx.device, dtype=torch.float32)
    flash_kda.fwd(
        q, k, v, raw_g, raw_beta,
        head_dim_qk ** -0.5 if scale is None else scale,
        attention, a_log, dt_bias, -5.0,
        initial_state=state_in, final_state=state_out,
    )
    _finish(attention[..., :head_dim_v], gate, norm_weight, o_proj, residual, out, norm_eps)
    final_state.copy_(state_out.transpose(-1, -2)[..., :head_dim_qk, :head_dim_v])
