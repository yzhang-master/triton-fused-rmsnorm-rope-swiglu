"""Rotary position embedding (half-split / "rotate_half" layout) for q and k in one launch.

For each position p and frequency index i < D/2, with (x1, x2) = (x[i], x[i + D/2]):

    out1 = x1 * cos(p, i) - x2 * sin(p, i)
    out2 = x2 * cos(p, i) + x1 * sin(p, i)

The map is a rotation, so the backward pass is the same kernel with sin negated: no
activation has to be saved beyond the (shared) cos/sin table. One program handles one
token and rotates every query *and* key head of it, so GQA models (fewer KV heads) cost
a single launch. With ``inplace=True`` the result overwrites the inputs; each element
pair is fully loaded before it is stored by the same program, which makes that safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import autotune, warp_configs


def build_rope_cache(
    max_positions: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 ``(cos, sin)`` tables of shape [max_positions, head_dim // 2]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
    return angles.cos().to(device).contiguous(), angles.sin().to(device).contiguous()


@triton.jit
def _rotate_heads(
    X,
    OUT,
    x_sb,
    x_ss,
    x_sh,
    o_sb,
    o_ss,
    o_sh,
    b,
    s,
    H,
    cos,
    sin,
    d,
    dmask,
    BLOCK_H: tl.constexpr,
    D_HALF: tl.constexpr,
):
    h = tl.arange(0, BLOCK_H)
    m = (h < H)[:, None] & dmask[None, :]
    src = X + b * x_sb + s * x_ss + h[:, None] * x_sh
    x1 = tl.load(src + d[None, :], mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(src + D_HALF + d[None, :], mask=m, other=0.0).to(tl.float32)
    o1 = x1 * cos[None, :] - x2 * sin[None, :]
    o2 = x2 * cos[None, :] + x1 * sin[None, :]
    dst = OUT + b * o_sb + s * o_ss + h[:, None] * o_sh
    tl.store(dst + d[None, :], o1.to(OUT.dtype.element_ty), mask=m)
    tl.store(dst + D_HALF + d[None, :], o2.to(OUT.dtype.element_ty), mask=m)


# restore_value: with inplace=True the kernel overwrites Q/K, which would otherwise be
# rotated once per autotune trial.
@autotune(warp_configs((1, 2, 4, 8)), key=["D_HALF"], restore_value=["Q", "K"])
@triton.jit
def _rope_kernel(
    Q,
    K,
    QO,
    KO,
    COS,
    SIN,
    POS,
    q_sb,
    q_ss,
    q_sh,
    k_sb,
    k_ss,
    k_sh,
    qo_sb,
    qo_ss,
    qo_sh,
    ko_sb,
    ko_ss,
    ko_sh,
    S,
    HQ,
    HK,
    D_HALF: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
    HAS_POS: tl.constexpr,
    INVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    if HAS_POS:
        pos = tl.load(POS + pid)
    else:
        pos = s
    d = tl.arange(0, BLOCK_D)
    dmask = d < D_HALF
    cos = tl.load(COS + pos * D_HALF + d, mask=dmask, other=1.0).to(tl.float32)
    sin = tl.load(SIN + pos * D_HALF + d, mask=dmask, other=0.0).to(tl.float32)
    if INVERSE:
        sin = -sin
    _rotate_heads(
        Q, QO, q_sb, q_ss, q_sh, qo_sb, qo_ss, qo_sh, b, s, HQ, cos, sin, d, dmask,
        BLOCK_HQ, D_HALF,
    )  # fmt: skip
    _rotate_heads(
        K, KO, k_sb, k_ss, k_sh, ko_sb, ko_ss, ko_sh, b, s, HK, cos, sin, d, dmask,
        BLOCK_HK, D_HALF,
    )  # fmt: skip


def _launch(q, k, cos, sin, positions, inplace, inverse):
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [batch, seq, heads, head_dim]")
    b, s, hq, d = q.shape
    if k.shape[:2] != (b, s) or k.shape[3] != d:
        raise ValueError("q and k must agree on batch, sequence and head_dim")
    if d % 2:
        raise ValueError("head_dim must be even")
    if cos.shape[-1] != d // 2 or sin.shape != cos.shape:
        raise ValueError("cos/sin must have shape [max_positions, head_dim // 2]")
    if inplace:
        if q.stride(-1) != 1 or k.stride(-1) != 1:
            raise ValueError("inplace=True needs a contiguous last dimension")
        qo, ko = q, k
    else:
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        qo, ko = torch.empty_like(q), torch.empty_like(k)
    hk = k.shape[2]
    pos = None
    if positions is not None:
        pos = positions.reshape(-1).to(torch.int32).contiguous()
        if pos.numel() != b * s:
            raise ValueError("positions must have shape [batch, seq]")
    cos, sin = cos.contiguous(), sin.contiguous()
    _rope_kernel[(b * s,)](
        q, k, qo, ko, cos, sin, pos if pos is not None else cos,
        *q.stride()[:3], *k.stride()[:3], *qo.stride()[:3], *ko.stride()[:3],
        s, hq, hk,
        D_HALF=d // 2,
        BLOCK_D=triton.next_power_of_2(d // 2),
        BLOCK_HQ=triton.next_power_of_2(hq),
        BLOCK_HK=triton.next_power_of_2(hk),
        HAS_POS=pos is not None,
        INVERSE=inverse,
    )  # fmt: skip
    return qo, ko


class RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, positions, inplace):
        qo, ko = _launch(q, k, cos, sin, positions, inplace, inverse=False)
        if inplace:
            ctx.mark_dirty(q, k)
        ctx.save_for_backward(cos, sin, positions)
        return qo, ko

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin, positions = ctx.saved_tensors
        dq, dk = _launch(dq, dk, cos, sin, positions, inplace=False, inverse=True)
        return dq, dk, None, None, None, None


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` [B, S, Hq, D] and ``k`` [B, S, Hk, D] with tables from ``build_rope_cache``.

    ``positions`` ([B, S] ints) selects table rows; by default row ``s`` is used for token
    ``s``. Positions are not range-checked (that would force a device sync).
    """
    return RopeFn.apply(q, k, cos, sin, positions, inplace)
