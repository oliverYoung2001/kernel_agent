"""Check output strides, causal boundaries and channel tails independently of either backend.

Run on a CUDA node from the repository root with
`python -m pytest -q -o consider_namespace_packages=true kernels/kda_megakernel/verification/tests/test_qkv_conv.py`.
"""
import pytest
import torch
import torch.nn.functional as F

from kernels.kda_megakernel.reference import convolve_qkv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA convolution")
@pytest.mark.parametrize("batch,lengths,hq,hv,dk,dv", [
    (2, None, 2, 4, 65, 33),
    (1, [2, 1, 16], 1, 2, 65, 95),
])
def test_single_conv_layout(batch, lengths, hq, hv, dk, dv):
    tokens = sum(lengths) if lengths else 19
    channels = 2 * hq * dk + hv * dv
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(123)
    merged = torch.randn((batch, tokens, channels + 7), device=device, dtype=torch.bfloat16, generator=generator)
    x = merged[..., :channels]
    weight = torch.randn((channels, 1, 4), device=device, dtype=x.dtype, generator=generator)
    offsets = torch.tensor([0, *lengths], device=device).cumsum(0) if lengths else None
    outputs = convolve_qkv(x, weight, offsets, hq, hv, dk, dv)

    chunks = x.split(lengths, dim=1) if lengths else [x]
    reference = []
    for chunk in chunks:
        padded = F.pad(chunk.transpose(1, 2).float(), (3, 0))
        conv = F.conv1d(padded, weight.float(), groups=channels)
        reference.append(F.silu(conv).transpose(1, 2).to(x.dtype))
    expected = torch.cat(reference, dim=1).split([hq * dk, hq * dk, hv * dv], dim=-1)
    assert len({output.untyped_storage().data_ptr() for output in outputs}) == 1
    for output, wanted in zip(outputs, expected, strict=True):
        assert output.is_contiguous()
        torch.testing.assert_close(output.flatten(2), wanted, rtol=.01, atol=.01)
