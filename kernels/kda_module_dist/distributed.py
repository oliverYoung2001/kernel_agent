"""Tensor contracts and token/head collectives for distributed KDA."""
from __future__ import annotations

import torch
import torch.distributed as dist

WEIGHT_NAMES = (
    "input_norm_weight", "in_proj_qkvbfg_a_weight", "conv1d_weight",
    "f_b_proj_weight", "g_b_proj_weight", "a_log_weight", "dt_bias_weight",
    "norm_weight", "o_proj_weight",
)


def topology(group=None):
    if dist.is_initialized():
        return dist.get_world_size(group), dist.get_rank(group)
    if group is not None:
        raise ValueError("A process group requires initialized torch.distributed")
    return 1, 0


def weight_shapes(hidden, hq, hv, dk, dv, width):
    c = 2 * hq * dk + hv * dv
    return dict(zip(WEIGHT_NAMES, (
        (hidden,), (hidden, c + hv + dk + dv), (c, 1, width),
        (dk, hv * dk), (dv, hv * dv), (hv,), (hv, dk), (dv,),
        (hv * dv, hidden),
    )))


def weight_indices(hq, hv, dk, dv, rank, world_size):
    """Select local contiguous head groups from each packed projection region."""
    if hq % world_size or hv % world_size:
        raise ValueError("TP/Ulysses require Hq and Hv divisible by the process-group size")
    q, v = hq * dk, hv * dv

    def part(start, size):
        local = size // world_size
        return list(range(start + rank * local, start + (rank + 1) * local))

    channels = part(0, q) + part(q, q) + part(2 * q, v)
    merged = channels + part(2 * q + v, hv) + list(range(2 * q + v + hv, 2 * q + v + hv + dk + dv))
    return {
        "in_proj_qkvbfg_a_weight": (1, merged), "conv1d_weight": (0, channels),
        "f_b_proj_weight": (1, part(0, hv * dk)),
        "g_b_proj_weight": (1, part(0, hv * dv)),
        "a_log_weight": (0, part(0, hv)), "dt_bias_weight": (0, part(0, hv)),
        "o_proj_weight": (0, part(0, hv * dv)),
    }


def partition_weights(weights, *, hq, hv, dk, dv, rank, world_size):
    """Materialize TP weights; retained mostly for independent verification."""
    selections = weight_indices(hq, hv, dk, dv, rank, world_size)
    result = {}
    for name, value in weights.items():
        if name in selections:
            axis, indices = selections[name]
            value = value.index_select(axis, torch.tensor(indices, device=value.device))
        result[name] = value.clone().contiguous()
    return result


def validate_input_specs(inpx, out, *, cu_seqlens, hq, hv, dk, dv,
                         weights, implementation, world_size=1):
    """Check local storage without allocating production-shaped intermediates."""
    if implementation not in ("reference", "tp", "ulysses", "kcp"):
        raise ValueError(f"Unknown implementation: {implementation}")
    if any(type(n) is not int or n < 1 for n in (hq, hv, dk, dv, world_size)) or hv % hq:
        raise ValueError("Positive head dimensions and Hv divisible by Hq are required")
    if inpx.ndim != 2 or inpx.shape[0] < 1 or inpx.shape[1] < 1:
        raise ValueError("inpx and out use contiguous [local_tokens, hidden] storage")
    if inpx.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Activations must use BF16, FP16 or FP32")
    if implementation != "reference" and inpx.dtype == torch.float32:
        raise ValueError("Native FLA baselines require BF16 or FP16 activations")
    if (cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int64
            or cu_seqlens.device != inpx.device or not cu_seqlens.is_contiguous()):
        raise ValueError("cu_seqlens must be contiguous global int64 boundaries on the input device")
    if cu_seqlens.device.type != "meta":
        ends = cu_seqlens.tolist()
        if ends[0] != 0 or ends[-1] != inpx.shape[0] * world_size or any(b <= a for a, b in zip(ends, ends[1:])):
            raise ValueError("Global sequence boundaries must cover all token shards exactly")
    if weights["conv1d_weight"].ndim != 3 or weights["conv1d_weight"].shape[-1] < 1:
        raise ValueError("conv1d_weight must have shape [channels, 1, width]")
    if implementation == "kcp" and dk > 256:
        raise ValueError("The native FLA CP backend requires Dk <= 256")
    if (implementation in ("ulysses", "kcp") and world_size > 1
            and inpx.shape[0] < weights["conv1d_weight"].shape[-1] - 1):
        raise ValueError("The native CP convolution needs at least W-1 tokens per rank")
    if implementation in ("tp", "ulysses") and (hq % world_size or hv % world_size):
        raise ValueError("TP/Ulysses require Hq and Hv divisible by the process-group size")
    local_hq, local_hv = (hq // world_size, hv // world_size) if implementation == "tp" else (hq, hv)
    shapes = weight_shapes(inpx.shape[1], local_hq, local_hv, dk, dv, weights["conv1d_weight"].shape[-1])
    tensors = {"inpx": inpx, "out": out, **weights}
    shapes.update(inpx=inpx.shape, out=inpx.shape)
    for name, tensor in tensors.items():
        dtype = torch.float32 if name in ("a_log_weight", "dt_bias_weight") else inpx.dtype
        if (tuple(tensor.shape) != tuple(shapes[name]) or tensor.dtype != dtype
                or tensor.device != inpx.device or not tensor.is_contiguous()):
            raise ValueError(f"{name}: expected contiguous {shapes[name]} {dtype} on {inpx.device}")
    if inpx.device.type != "meta":
        output_storage = out.untyped_storage().data_ptr()
        if any(output_storage == value.untyped_storage().data_ptr() for value in (inpx, cu_seqlens, *weights.values())):
            raise ValueError("out must not alias inputs or weights")


def gather_tokens(value, group=None):
    p, _ = topology(group)
    if p == 1:
        return value
    result = torch.empty((p * value.shape[0], *value.shape[1:]), device=value.device, dtype=value.dtype)
    dist.all_gather_into_tensor(result, value.contiguous(), group=group)
    return result


def scatter_sum(value, out, group=None):
    p, _ = topology(group)
    if p == 1:
        out.copy_(value)
    else:
        dist.reduce_scatter_tensor(out, value.contiguous(), group=group)


def sequence_to_head(value, group=None):
    """[1,L,H,D...] -> [1,P*L,H/P,D...] in global token order."""
    p, _ = topology(group)
    if value.ndim < 3 or value.shape[0] != 1 or value.shape[2] % p:
        raise ValueError("Expected [1,L,H,...] with H divisible by world size")
    if p == 1:
        return value.contiguous()
    _, length, heads, *tail = value.shape
    send = value.reshape(length, p, heads // p, *tail).transpose(0, 1).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.reshape(1, p * length, heads // p, *tail)


def head_to_sequence(value, group=None):
    """Invert sequence_to_head, concatenating source ranks in head order."""
    p, _ = topology(group)
    if value.ndim < 3 or value.shape[0] != 1 or value.shape[1] % p:
        raise ValueError("Expected [1,P*L,Hlocal,...]")
    if p == 1:
        return value.contiguous()
    _, tokens, heads, *tail = value.shape
    send = value.reshape(p, tokens // p, heads, *tail).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return recv.transpose(0, 1).reshape(1, tokens // p, p * heads, *tail).contiguous()
