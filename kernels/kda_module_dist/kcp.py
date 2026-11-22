"""Native KDA CP API and the complete-module projection adapter.

``chunk_kda`` and ``build_cp_context`` are the actual FLA public functions,
including their signatures, dispatch, autograd and validation. There is no
second implementation of the CP affine scan or its communication kernels.
"""
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda

__all__ = ["build_cp_context", "chunk_kda", "module_recurrence"]


def module_recurrence(q, k, v, decay, beta_logits, a_log, dt_bias, offsets, context=None):
    """Adapt raw module projections to the call used by tests/cp/test_kda.py.

    Q/K/V and raw gate keep their activation dtype and layout. Beta sigmoid
    is stored in the activation dtype, as in the upstream test's input maker.
    The public core API instead accepts already activated beta by default.
    """
    options = {"cu_seqlens": offsets} if context is None else {"cp_context": context}
    return chunk_kda(
        q=q, k=k, v=v, g=decay, beta=beta_logits.sigmoid(),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0,
        disable_recompute=False, state_v_first=False,
        A_log=a_log, dt_bias=dt_bias.flatten(), **options,
    )[0]
