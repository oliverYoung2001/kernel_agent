"""Full KDA attention with an FP32 token recurrence as the numerical oracle."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl

L2_EPS = 1e-6
RMS_EPS = 1e-5
LOWER_BOUND = -5.0


def _buffers(inpx, out):
    """Expose packed storage and sequence boundaries without copying tensors."""
    if inpx.is_nested != out.is_nested:
        raise ValueError("inpx and out must both be dense or both be jagged")
    if not inpx.is_nested:
        return inpx, out, None
    if inpx.layout != torch.jagged or out.layout != torch.jagged:
        raise ValueError("Packed tensors must use the jagged layout")
    offsets = inpx.offsets()
    if out.offsets() is not offsets or inpx.lengths() is not None or out.lengths() is not None:
        raise ValueError("Packed tensors must share offsets and have no holes")
    return inpx.values().unsqueeze(0), out.values().unsqueeze(0), offsets


def validate_input_specs(
    inpx, out, final_state, *, input_norm_weight, in_proj_qkvbfg_a_weight,
    conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
    dt_bias_weight, norm_weight, o_proj_weight,
):
    """Infer head geometry from the convolution and final state."""
    x, y, offsets = _buffers(inpx, out)
    if x.ndim != 3 or final_state.ndim != 4 or conv1d_weight.ndim != 3:
        raise ValueError("Expected inpx[B,S,M], final_state[N,Hv,Dk,Dv] and convolution[C,1,W]")
    b, s, m = x.shape
    n, hv, dk, dv = final_state.shape
    c, groups, width = conv1d_weight.shape
    hq, remainder = divmod(c - hv * dv, 2 * max(1, dk))
    expected_n = b if offsets is None else offsets.numel() - 1
    if (n != expected_n or min(b, s, m, hq, hv, dk, dv, width, n) < 1
            or groups != 1 or remainder or hv % hq):
        raise ValueError("Inconsistent batch, convolution or head geometry")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Expected BF16, FP16 or FP32 activations")
    if offsets is not None and (offsets.dtype != torch.int64 or offsets.device != x.device
                                or not offsets.is_contiguous()):
        raise ValueError("Packed offsets must be contiguous int64 on the input device")
    act = x.dtype
    specs = {
        "inpx": (x, (b, s, m), act),
        "out": (y, (b, s, m), act),
        "final_state": (final_state, (n, hv, dk, dv), torch.float32),
        "input_norm_weight": (input_norm_weight, (m,), act),
        "in_proj_qkvbfg_a_weight": (in_proj_qkvbfg_a_weight, (m, c + hv + dk + dv), act),
        "conv1d_weight": (conv1d_weight, (c, 1, width), act),
        "f_b_proj_weight": (f_b_proj_weight, (dk, hv * dk), act),
        "g_b_proj_weight": (g_b_proj_weight, (dv, hv * dv), act),
        "a_log_weight": (a_log_weight, (hv,), torch.float32),
        "dt_bias_weight": (dt_bias_weight, (hv, dk), torch.float32),
        "norm_weight": (norm_weight, (dv,), act),
        "o_proj_weight": (o_proj_weight, (hv * dv, m), act),
    }
    for name, (tensor, shape, dtype) in specs.items():
        if (tensor.shape != shape or tensor.dtype != dtype
                or tensor.device != x.device or not tensor.is_contiguous()):
            raise ValueError(f"{name} must be contiguous {shape} {dtype} on {x.device}")
    return x, y, offsets, (hq, hv, dk, dv)


def validate_arguments(inpx, out, final_state, **weights):
    """Validate the CUDA buffers and return their kernel views and head geometry."""
    assert inpx.is_cuda, "KDA attention requires CUDA tensors"
    x, y, offsets, heads = validate_input_specs(inpx, out, final_state, **weights)
    inputs = {tensor.untyped_storage().data_ptr() for tensor in (x, *weights.values())}
    outputs = {tensor.untyped_storage().data_ptr() for tensor in (y, final_state)}
    if len(outputs) != 2 or outputs & inputs:
        raise ValueError("Output buffers must not alias inputs or each other")
    return x, y, offsets, heads


@tr.jit
def _conv_qkv(X, W, Y, Seq, B: tl.constexpr, S: tl.constexpr,
              CQ: tl.constexpr, CV: tl.constexpr, WIDTH: tl.constexpr,
              SB: tl.constexpr, SS: tl.constexpr,
              TM: tl.constexpr, TN: tl.constexpr, PACKED: tl.constexpr):
    plane = tl.program_id(2).to(tl.int64) // B
    sample = tl.program_id(2).to(tl.int64) % B
    channel_start = plane * CQ
    C = tl.where(plane == 2, CV, CQ)
    # Each output region is contiguous across batches, tokens, heads and features.
    Y += channel_start * B * S
    t = (tl.program_id(0) * TM + tl.arange(0, TM)).to(tl.int64)
    c = tl.program_id(1) * TN + tl.arange(0, TN)
    if PACKED:
        seq = tl.load(Seq + t, t < S, -1)
    acc = tl.full((TM, TN), 0, tl.float32)
    for tap in tl.static_range(WIDTH):
        p = t + tap - WIDTH + 1
        valid = (p >= 0) & (t < S)
        if PACKED:
            valid &= tl.load(Seq + p, valid, -2) == seq
        a = tl.load(X + sample * SB + p[:, None] * SS + channel_start + c[None, :],
                    valid[:, None] & (c[None, :] < C), 0).to(tl.float32)
        w = tl.load(W + (channel_start + c) * WIDTH + tap, c < C, 0).to(tl.float32)
        acc += a * w[None, :]
    tl.store(Y + (sample * S + t[:, None]) * C + c[None, :], acc * tl.sigmoid(acc),
             (t[:, None] < S) & (c[None, :] < C))


def convolve_qkv(qkv, weight, offsets, hq, hv, dk, dv):
    b, s, _ = qkv.shape
    seq = None
    if offsets is not None:
        seq = torch.searchsorted(offsets[1:], torch.arange(s, device=qkv.device), right=True, out_int32=True)
    cq, cv = hq * dk, hv * dv
    q_size, v_size = b * s * cq, b * s * cv
    storage = torch.empty(2 * q_size + v_size, device=qkv.device, dtype=qkv.dtype)
    _conv_qkv[(tr.cdiv(s, 16), tr.cdiv(max(cq, cv), 128), 3 * b)](
        qkv, weight, storage, seq, b, s, cq, cv, weight.shape[2], *qkv.stride()[:2],
        16, 128, offsets is not None, enable_fp_fusion=False)
    q, k, v = storage.split((q_size, q_size, v_size))
    return (q.view(b, s, hq, dk), k.view(b, s, hq, dk), v.view(b, s, hv, dv))


@tr.jit
def _wide_causal_silu(
    X, W, Y, Seq, S: tl.constexpr, C: tl.constexpr, Width: tl.constexpr,
    SB: tl.constexpr, SS: tl.constexpr, TM: tl.constexpr, TN: tl.constexpr, PACKED: tl.constexpr,
):
    sample = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * TM + tl.arange(0, TM)).to(tl.int64)
    c = (tl.program_id(1) * TN + tl.arange(0, TN)).to(tl.int64)
    if PACKED:
        sequence = tl.load(Seq + t, t < S, -1)
    accum = tl.full((TM, TN), 0, tl.float32)
    for lag in tl.static_range(Width):
        past = t + lag - Width + 1
        valid = (t < S) & (past >= 0)
        if PACKED:
            valid &= tl.load(Seq + past, valid, -2) == sequence
        value = tl.load(X + sample * SB + past[:, None] * SS + c[None, :],
                        valid[:, None] & (c[None, :] < C), 0).to(tl.float32)
        weight = tl.load(W + c * Width + lag, c < C, 0).to(tl.float32)
        accum += value * weight[None, :]
    tl.store(Y + (sample * S + t[:, None]) * C + c[None, :],
             accum * tl.sigmoid(accum), (t[:, None] < S) & (c[None, :] < C))


def _causal_silu(qkv, weight, offsets):
    channels, _, width = weight.shape
    if channels % 8 == 0 and width in (2, 3, 4):
        from causal_conv1d.cpp_functions import _causal_conv1d_fwd_cpp

        seq_idx = None
        if offsets is not None:
            tokens = torch.arange(qkv.shape[1], device=qkv.device)
            seq_idx = torch.searchsorted(offsets[1:], tokens, right=True, out_int32=True)[None]
        out = torch.empty(qkv.shape, device=qkv.device, dtype=qkv.dtype)
        max_index = sum((size - 1) * stride for size, stride in zip(qkv.shape, qkv.stride()))
        if max_index >= 2**32 or max(qkv.stride()) >= 2**32:
            # The installed causal_conv1d extension uses uint32 element offsets.
            b, s, c = qkv.shape
            _wide_causal_silu[(tr.cdiv(s, 16), tr.cdiv(c, 128), b)](
                qkv, weight, out, seq_idx, s, c, width, *qkv.stride()[:2], 16, 128,
                seq_idx is not None, enable_fp_fusion=False,
            )
            return out
        if qkv.stride(1) % 8:
            qkv = qkv.contiguous()
        _causal_conv1d_fwd_cpp(qkv.transpose(1, 2), weight.squeeze(1), None,
                             seq_idx, None, out.transpose(1, 2), None, True)
        return out

    def convolve(x):
        y = F.conv1d(x.transpose(1, 2).float(), weight.float(),
                     padding=width - 1, groups=channels)
        return F.silu(y[..., :x.shape[1]]).transpose(1, 2).to(x.dtype)

    if offsets is None:
        return convolve(qkv).contiguous()
    ends = offsets.tolist()
    return torch.cat([convolve(qkv[:, a:b]) for a, b in zip(ends, ends[1:])], dim=1)


@tr.jit
def _normalize_packed_view(X, Y, ROWS: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                           SB: tl.constexpr, SS: tl.constexpr, SH: tl.constexpr, BS: tl.constexpr):
    row = (tl.program_id(0) * BS + tl.arange(0, BS)).to(tl.int64)
    d = tl.arange(0, 128)
    offset = row // (S * H) * SB + row // H % S * SS + row % H * SH
    value = tl.load(X + offset[:, None] + d[None, :], row[:, None] < ROWS, 0).to(tl.float32)
    normalized = value * tl.rsqrt(tl.sum(value * value, 1) + 1e-6)[:, None]
    tl.store(Y + row[:, None] * 128 + d[None, :], normalized, row[:, None] < ROWS)


def normalize_qk(x):
    if x.shape[-1] == 128:
        b, s, h, _ = x.shape
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        _normalize_packed_view[(tr.cdiv(b * s * h, 32),)](
            x, out, b * s * h, s, h, *x.stride()[:3], 32,
        )
        return out
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + L2_EPS)).to(x.dtype).contiguous()


def _project(x, input_norm, merged_weight, conv_weight, decay_weight, hq, hv, dk, dv, offsets, convolution):
    from fla.modules.layernorm import rms_norm

    normalized = rms_norm(x, input_norm, None, eps=RMS_EPS)
    c = conv_weight.shape[0]
    merged = normalized @ merged_weight
    qkv, beta, f_a, g_a = merged.split((c, hv, dk, dv), dim=-1)
    decay = (f_a @ decay_weight).unflatten(-1, (hv, dk))
    q, k, v = convolution(qkv, conv_weight, offsets, hq, hv, dk, dv)
    return q, k, v, decay, beta.contiguous(), g_a.contiguous()


def _reference_convolution(qkv, weight, offsets, hq, hv, dk, dv):
    conv = _causal_silu(qkv, weight, offsets)
    q, k, v = conv.split((hq * dk, hq * dk, hv * dv), dim=-1)
    return q.unflatten(-1, (hq, dk)), k.unflatten(-1, (hq, dk)), v.unflatten(-1, (hv, dv))


def _attention(
    recurrence, inpx, out, final_state, *, convolution=convolve_qkv, input_norm_weight, in_proj_qkvbfg_a_weight,
    conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
    dt_bias_weight, norm_weight, o_proj_weight,
):
    x, y, offsets, (hq, hv, dk, dv) = validate_arguments(
        inpx, out, final_state, input_norm_weight=input_norm_weight,
        in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight, conv1d_weight=conv1d_weight,
        f_b_proj_weight=f_b_proj_weight, g_b_proj_weight=g_b_proj_weight,
        a_log_weight=a_log_weight, dt_bias_weight=dt_bias_weight,
        norm_weight=norm_weight, o_proj_weight=o_proj_weight,
    )
    q, k, v, decay, beta, g_a = _project(
        x, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
        f_b_proj_weight, hq, hv, dk, dv, offsets, convolution,
    )
    attention = recurrence(q, k, v, decay, beta, a_log_weight, dt_bias_weight, final_state, offsets)
    del q, k, v, decay, beta
    gate = (g_a @ g_b_proj_weight).view(-1, dv)
    attention = attention.reshape(-1, dv)
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    gated = layer_norm_gated_fwd(
        attention, gate, norm_weight, None, activation="sigmoid",
        eps=RMS_EPS, is_rms_norm=True,
    )[0]
    torch.mm(gated.view(-1, hv * dv), o_proj_weight, out=y.view(-1, y.shape[-1]))
    y.add_(x)


@tr.jit
def _token_recurrence(
    Q, K, V, A, Beta, O, Final, Offsets, S: tl.constexpr, HQ: tl.constexpr, HV: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    SCALE: tl.constexpr, PACKED: tl.constexpr,
):
    sample_head, part = tl.program_id(0), tl.program_id(1)
    sample, head = sample_head // HV, sample_head % HV
    qhead = head // (HV // HQ)
    if PACKED:
        start, end = tl.load(Offsets + sample), tl.load(Offsets + sample + 1)
    else:
        start, end = sample * S, (sample + 1) * S
    key = tl.arange(0, BK)
    value = part * BV + tl.arange(0, BV)
    state = tl.full((BK, BV), 0, tl.float32)
    for t in range(start, end):
        qk = (t * HQ + qhead) * DK + key
        vh = (t * HV + head) * DV + value
        q = tl.load(Q + qk, key < DK, 0).to(tl.float32)
        k = tl.load(K + qk, key < DK, 0).to(tl.float32)
        v = tl.load(V + vh, value < DV, 0).to(tl.float32)
        alpha = tl.load(A + (t * HV + head) * DK + key, key < DK, 0)
        beta = tl.load(Beta + t * HV + head)
        state = state * tl.exp(alpha)[:, None]
        state += k[:, None] * (beta * (v - tl.sum(k[:, None] * state, 0)))[None, :]
        tl.store(O + vh, SCALE * tl.sum(q[:, None] * state, 0), value < DV)
    tl.store(Final + sample_head * DK * DV + key[:, None] * DV + value[None, :],
             state, (key[:, None] < DK) & (value[None, :] < DV))


def _reference_recurrence(q, k, v, decay, beta, a_log, dt_bias, final, offsets):
    q, k, v = normalize_qk(q), normalize_qk(k), v.contiguous()
    alpha = decay.float().add(dt_bias).mul_(a_log.exp()[:, None]).sigmoid_().mul_(LOWER_BOUND)
    beta = beta.float().sigmoid()
    n, hv, dk, dv = final.shape
    scale = dk ** -0.5
    out = torch.empty_like(v)
    bv = min(32, tr.next_power_of_2(dv))
    _token_recurrence[(n * hv, tr.cdiv(dv, bv))](
        q, k, v, alpha, beta, out, final, offsets, v.shape[1], q.shape[2], hv,
        dk, dv, tr.next_power_of_2(dk), bv, scale, offsets is not None,
    )
    return out


@torch.no_grad()
def reference(
    inpx, out, final_state, *, input_norm_weight, in_proj_qkvbfg_a_weight,
    conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
    dt_bias_weight, norm_weight, o_proj_weight,
):
    """Apply full attention to dense or packed activations with zero initial state."""
    _attention(
        _reference_recurrence, inpx, out, final_state, convolution=_reference_convolution, input_norm_weight=input_norm_weight,
        in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight, conv1d_weight=conv1d_weight,
        f_b_proj_weight=f_b_proj_weight, g_b_proj_weight=g_b_proj_weight,
        a_log_weight=a_log_weight, dt_bias_weight=dt_bias_weight,
        norm_weight=norm_weight, o_proj_weight=o_proj_weight,
    )
