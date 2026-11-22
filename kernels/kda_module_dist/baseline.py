"""TP, Ulysses SP and native FLA KCP baselines for the complete KDA module.

Tensor metadata (static-derived; strides are in elements, not bytes)
=================================================================
Symbols use the GLOBAL head counts passed to the public baseline functions:
    P = process-group size; L = local tokens; T = P*L; M = hidden size;
    Hq = hq; Hv = hv; K = dk; V = dv; W = convolution width;
    N = number of global sequences; C = 2*Hq*K + Hv*V;
    A = inpx.dtype, either torch.bfloat16 or torch.float16.
    Cj = 2*Q*K + H*V; F = Cj + H + K + V, where (J,Q,H) below are
    the token count and head counts used by _project and _finish.

Implementation       J       Q          H         Cj
    tp               T       Hq/P       Hv/P      C/P
    ulysses          L       Hq         Hv        C
    kcp              L       Hq         Hv        C

The tables give canonical row-major strides. Validation requires contiguous
inputs, but PyTorch permits arbitrary strides on size-one axes; views and
.contiguous() may retain those strides. Thus singleton-axis strides can differ
without changing any addressed element. No autocast is assumed. All tensors
are on inpx.device (CUDA), except the explicitly noted CPU context metadata.
The scope is tensors passed/created in this file and _project's returned views,
not private workspaces inside FLA kernels. KCP's core API is exported in kcp.py;
its module adapter uses the fused call in FLA tests/cp/test_kda.py.

Public tensor arguments (``weights`` also references the nine weight tensors):
    name                         shape                  dtype       stride
    inpx, out                    (L,M)                  A           (M,1)
    input_norm_weight            (M,)                   A           (1,)
    in_proj_qkvbfg_a_weight       (M,F)                  A           (F,1)
    conv1d_weight                (Cj,1,W)               A           (W,W,1)
    f_b_proj_weight              (K,H*K)                A           (H*K,1)
    g_b_proj_weight              (V,H*V)                A           (H*V,1)
    a_log_weight                 (H,)                   float32     (1,)
    dt_bias_weight               (H,K)                  float32     (K,1)
    norm_weight                  (V,)                   A           (1,)
    o_proj_weight                (H*V,M)                A           (M,1)
    cu_seqlens                   (N+1,)                 int64       (1,)
TP receives head-sharded weights; Ulysses/KCP receive full weights. The input
and output token shards always have L rows, including TP. cu_seqlens always
describes GLOBAL token boundaries [0,...,T], including packed sequences.

Projection metadata, common formulas with implementation-specific J,Q,H,Cj,F:
    tensor                       shape                  dtype       stride
    x, normalized in _project    (1,J,M)                A           (J*M,M,1)
    merged in _project           (1,J,F)                A           (J*F,F,1)
    qkv passed to convolution    (1,J,Cj)               A           (J*F,F,1)
    beta split in _project       (1,J,H)                A           (J*F,F,1)
    f_a split in _project        (1,J,K)                A           (J*F,F,1)
    g_a split in _project        (1,J,V)                A           (J*F,F,1)
    f_a @ f_b_proj_weight        (1,J,H*K)              A           (J*H*K,H*K,1)
    decay returned by _project   (1,J,H,K)              A           (J*H*K,H*K,K,1)
    beta returned by _project    (1,J,H)                A           (J*H,H,1)
    g_a returned by _project     (1,J,V)                A           (J*V,V,1)
The split tensors share merged storage with different storage offsets; stride
does not encode that offset. _project materializes beta and g_a via contiguous.

Q/K/V returned by _project (all dtype A):
    tp, or P=1 for any implementation: convolve_qkv uses separate packed
        regions, so q/k: (1,J,Q,K), stride (J*Q*K,Q*K,K,1);
                    v: (1,J,H,V), stride (J*H*V,H*V,V,1).
    ulysses/kcp with P>1: the nested CP convolution returns contiguous
        conv: (1,L,C), stride (L*C,C,1), then splits channel views:
        q/k: (1,L,Hq,K), stride (L*C,C,K,1);
          v: (1,L,Hv,V), stride (L*C,C,V,1).
        q/k/v storage offsets in conv are 0, Hq*K, 2*Hq*K respectively.

Implementation-specific recurrence and output metadata (all dtype A here):
    tp:
        gather_tokens(inpx): (T,M), stride (M,1); x is its unsqueeze view.
        _recurrence receives (J,Q,H)=(T,Hq/P,Hv/P).
        attention: (1,T,Hv/P,V), stride (T*(Hv/P)*V,(Hv/P)*V,V,1).
        partial: (T,M), stride (M,1); reduce-scatter SUM writes out (L,M).
    ulysses:
        x = inpx.unsqueeze(0): (1,L,M), stride (L*M,M,1).
        sequence_to_head replaces each of the following with a contiguous tensor:
            q/k:   (1,T,Hq/P,K), stride (T*(Hq/P)*K,(Hq/P)*K,K,1)
            v:     (1,T,Hv/P,V), stride (T*(Hv/P)*V,(Hv/P)*V,V,1)
            decay: (1,T,Hv/P,K), stride (T*(Hv/P)*K,(Hv/P)*K,K,1)
            beta:  (1,T,Hv/P),   stride (T*(Hv/P),Hv/P,1)
        g_a stays (1,L,V); it has no head axis and is not exchanged.
        Sliced a_log_weight: (Hv/P,), float32, stride (1,).
        Sliced dt_bias_weight: (Hv/P,K), float32, stride (K,1).
        _recurrence returns attention (1,T,Hv/P,V), with the same stride as v.
        head_to_sequence replaces attention with (1,L,Hv,V),
            stride (L*Hv*V,Hv*V,V,1), before _finish uses all heads.
    kcp:
        x = inpx.unsqueeze(0): (1,L,M), stride (L*M,M,1).
        _recurrence receives (J,Q,H)=(L,Hq,Hv), using the projection layouts.
        attention: (1,L,Hv,V), stride (L*Hv*V,Hv*V,V,1).
        _finish writes out (L,M) directly, as in Ulysses.

For P>1, Ulysses/KCP build an FLACPContext object for convolution halo exchange.
It is not a tensor. Its cu_seqlens / cu_seqlens_cpu tensors use rank-local
boundaries: (Nr+1,), int32, stride (1,), on CUDA / CPU respectively, where Nr
counts sequence fragments intersecting the rank. Ulysses passes GLOBAL int64
cu_seqlens to recurrence after all-to-all; KCP passes context instead (FLA uses
its LOCAL int32 boundaries). For P=1, context=None and global offsets are local.
"""
from __future__ import annotations

import inspect

import torch
import torch.distributed as dist

from .distributed import (
    gather_tokens, head_to_sequence, scatter_sum, sequence_to_head, topology,
    validate_input_specs,
)


def _recurrence(q, k, v, decay, beta, a_log, dt_bias, offsets, context=None):
    """Metadata at the KDA boundary; J,Q,H are the recurrence-local dimensions.

    q/k: (1,J,Q,K), A; v: (1,J,H,V), A; input strides are listed above.
    decay: (1,J,H,K), A, stride (J*H*K,H*K,K,1).
    beta: (1,J,H), A, stride (J*H,H,1).
    a_log: (H,), float32, stride (1,); dt_bias: (H,K), float32, (K,1).
    offsets aliases global cu_seqlens: (N+1,), int64, stride (1,).
    This helper is used by TP/Ulysses only. KCP uses kcp.module_recurrence.

    alpha: (1,J,H,K), float32, stride (J*H*K,H*K,K,1), NOT activation dtype.
    normalize_qk(q/k): (1,J,Q,K), A, stride (J*Q*K,Q*K,K,1).
    v.contiguous(): (1,J,H,V), A, stride (J*H*V,H*V,V,1).
    beta.float() and its sigmoid: (1,J,H), float32, stride (J*H,H,1).
    options is a dict holding offsets or context; it creates no tensor.
    chunk_kda returns (attention, None); attention: (1,J,H,V), A,
    stride (J*H*V,H*V,V,1). output_final_state=False exposes no state tensor.
    """
    from fla.ops.kda import chunk_kda
    from fla.ops.kda.gate import kda_gate_fwd
    from kernels.kda_megakernel.reference import LOWER_BOUND, normalize_qk

    # Gate activations and beta sigmoid are FP32 even when Q/K/V are BF16/FP16.
    alpha = kda_gate_fwd(decay, a_log, dt_bias, lower_bound=LOWER_BOUND)
    options = {"cu_seqlens": offsets} if context is None else {"cp_context": context}
    return inspect.unwrap(chunk_kda)(
        q=normalize_qk(q), k=normalize_qk(k), v=v.contiguous(), g=alpha,
        beta=beta.float().sigmoid(), scale=q.shape[-1] ** -0.5,
        safe_gate=True, output_final_state=False, **options,
    )[0]


def _finish(attention, g_a, g_b, norm, o_proj, out):
    """Output metadata; J,H are (T,Hv/P) for TP and (L,Hv) otherwise.

    All tensors below have dtype A. V=dv, M=hidden; local hv below equals H.
    attention: (1,J,H,V), stride (J*H*V,H*V,V,1).
    g_a: (1,J,V), stride (J*V,V,1).
    g_b aliases g_b_proj_weight: (V,H*V), stride (H*V,1).
    norm aliases norm_weight: (V,), stride (1,).
    o_proj aliases o_proj_weight: (H*V,M), stride (M,1).
    g_a @ g_b: (1,J,H*V), stride (J*H*V,H*V,1).
    gate and attention.reshape(-1,V): (J*H,V), stride (V,1).
    gated: (J*H,V), stride (V,1).
    gated.view(-1,H*V): (J,H*V), stride (H*V,1).
    out: (J,M), stride (M,1), aliasing partial for TP or public out otherwise.
    torch.mm writes this buffer; no separately allocated mm result is retained.
    """
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd
    from kernels.kda_megakernel.reference import RMS_EPS

    hv, dv = attention.shape[-2:]
    gate = (g_a @ g_b).reshape(-1, dv)
    gated = layer_norm_gated_fwd(
        attention.reshape(-1, dv), gate, norm, None, activation="sigmoid",
        eps=RMS_EPS, is_rms_norm=True,
    )[0]
    torch.mm(gated.view(-1, hv * dv), o_proj, out=out)


def _execute(implementation, inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight,
             conv1d_weight, f_b_proj_weight, g_b_proj_weight, a_log_weight,
             dt_bias_weight, norm_weight, o_proj_weight, cu_seqlens, hq, hv, dk, dv, group):
    from kernels.kda_megakernel.reference import _project, convolve_qkv

    p, rank = topology(group)
    weights = dict(input_norm_weight=input_norm_weight, in_proj_qkvbfg_a_weight=in_proj_qkvbfg_a_weight,
                   conv1d_weight=conv1d_weight, f_b_proj_weight=f_b_proj_weight,
                   g_b_proj_weight=g_b_proj_weight, a_log_weight=a_log_weight,
                   dt_bias_weight=dt_bias_weight, norm_weight=norm_weight, o_proj_weight=o_proj_weight)
    validate_input_specs(inpx, out, cu_seqlens=cu_seqlens, hq=hq, hv=hv, dk=dk, dv=dv,
                         weights=weights, implementation=implementation, world_size=p)
    if not inpx.is_cuda:
        raise ValueError("The production baselines require CUDA")
    context = None
    if implementation == "tp":
        # x: (1,T,M), A, stride (T*M,M,1); weights already contain local heads.
        x = gather_tokens(inpx, group).unsqueeze(0)
        hq, hv = hq // p, hv // p
        convolution = convolve_qkv
    else:
        # Ulysses/KCP x: (1,L,M), A, stride (L*M,M,1); weights contain all heads.
        x = inpx.unsqueeze(0)
        convolution = convolve_qkv
        if p > 1:
            from fla.ops.cp import build_cp_context
            # from fla.modules.convolution import causal_conv1d
            from fla.modules.causal_conv1d import causal_conv1d

            context = build_cp_context(cu_seqlens, group=dist.group.WORLD if group is None else group,
                                       conv1d_kernel_size=conv1d_weight.shape[-1])

            def convolution(qkv, weight, offsets, hq, hv, dk, dv):
                # Only Ulysses/KCP with P>1 reach this closure. All data dtype A.
                # qkv: (1,L,C), stride (L*F,F,1), F=C+Hv+K+V (a merged view).
                # weight: (C,1,W), stride (W,W,1); squeeze(1): (C,W), (W,1).
                # offsets aliases global cu_seqlens: (N+1,), int64, (1,);
                # causal_conv1d uses context's local boundaries instead.
                # The CP convolution exchanges the causal halo before _project returns.
                conv, _ = causal_conv1d(x=qkv, weight=weight.squeeze(1), activation="silu", cp_context=context)
                # conv: (1,L,C), stride (L*C,C,1); discarded final state is None.
                q, k, v = conv.split((hq * dk, hq * dk, hv * dv), dim=-1)
                # Split q/k: (1,L,Hq*K), v: (1,L,Hv*V), all stride (L*C,C,1).
                # Unflatten preserves token pitch C: q/k (L*C,C,K,1), v (L*C,C,V,1).
                return q.unflatten(-1, (hq, dk)), k.unflatten(-1, (hq, dk)), v.unflatten(-1, (hv, dv))

    # Returned layouts: see the projection and Q/K/V tables in the module docstring.
    q, k, v, decay, beta, g_a = _project(
        x, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
        f_b_proj_weight, hq, hv, dk, dv, cu_seqlens, convolution,
    )
    if implementation == "ulysses":
        # Each tensor becomes contiguous with T tokens and 1/P of its own heads.
        q, k, v, decay, beta = (sequence_to_head(tensor, group) for tensor in (q, k, v, decay, beta))
        local_hv = hv // p
        # FP32: (Hv/P,), stride (1,); (Hv/P,K), stride (K,1). Head slices only.
        a_log_weight = a_log_weight[rank * local_hv:(rank + 1) * local_hv].contiguous()
        dt_bias_weight = dt_bias_weight[rank * local_hv:(rank + 1) * local_hv].contiguous()
        attention = _recurrence(q, k, v, decay, beta, a_log_weight, dt_bias_weight, cu_seqlens)
        # (1,T,Hv/P,V) -> (1,L,Hv,V), A; result contiguous for the local output.
        attention = head_to_sequence(attention, group)
    elif implementation == "kcp":
        from .kcp import module_recurrence

        attention = module_recurrence(q, k, v, decay, beta, a_log_weight, dt_bias_weight,
                                      cu_seqlens, context)
    else:
        attention = _recurrence(q, k, v, decay, beta, a_log_weight, dt_bias_weight,
                                cu_seqlens)
    del q, k, v, decay, beta
    if implementation == "tp":
        # partial: (T,M), A, stride (M,1); holds this rank's head contribution.
        partial = torch.empty((x.shape[1], out.shape[1]), device=out.device, dtype=out.dtype)
        _finish(attention, g_a, g_b_proj_weight, norm_weight, o_proj_weight, partial)
        scatter_sum(partial, out, group)
    else:
        _finish(attention, g_a, g_b_proj_weight, norm_weight, o_proj_weight, out)
    # All implementations: out/inpx (L,M), A, stride (M,1); residual is in-place.
    out.add_(inpx)


@torch.no_grad()
def baseline_tp(
    inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
    f_b_proj_weight, g_b_proj_weight, a_log_weight, dt_bias_weight, norm_weight,
    o_proj_weight, cu_seqlens, hq, hv, dk, dv, group=None,
):
    """Token all-gather, local head computation, output reduce-scatter and residual.

    Tensor shapes/dtypes/strides: module tables, implementation="tp".
    Inputs/weights are read-only; out (L,M), dtype A, stride (M,1) is written.
    """
    _execute("tp", **locals())


@torch.no_grad()
def baseline_ulysses(
    inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
    f_b_proj_weight, g_b_proj_weight, a_log_weight, dt_bias_weight, norm_weight,
    o_proj_weight, cu_seqlens, hq, hv, dk, dv, group=None,
):
    """Local projections, token/head all-to-all, KDA, inverse all-to-all and output.

    Tensor shapes/dtypes/strides: module tables, implementation="ulysses".
    Inputs/weights are read-only; out (L,M), dtype A, stride (M,1) is written.
    """
    _execute("ulysses", **locals())


@torch.no_grad()
def baseline_kcp(
    inpx, out, *, input_norm_weight, in_proj_qkvbfg_a_weight, conv1d_weight,
    f_b_proj_weight, g_b_proj_weight, a_log_weight, dt_bias_weight, norm_weight,
    o_proj_weight, cu_seqlens, hq, hv, dk, dv, group=None,
):
    """Local token computation with FLA's affine-state all-gather and merge.

    Tensor shapes/dtypes/strides: module tables, implementation="kcp".
    Inputs/weights are read-only; out (L,M), dtype A, stride (M,1) is written.
    """
    _execute("kcp", **locals())
