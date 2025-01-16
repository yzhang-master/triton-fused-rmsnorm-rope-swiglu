"""RMSNorm and fused residual-add + RMSNorm (forward and backward) in Triton.

    y = x * rstd * w,   rstd = 1 / sqrt(mean(x^2) + eps)

Backward, with x_hat = x * rstd and g = dy * w:

    dx = rstd * (g - x_hat * mean(g * x_hat))
    dw = sum_rows(dy * x_hat)

One program handles one row, so the row is read from HBM once in forward and once in
backward. ``rstd`` (4 bytes per row) is the only activation saved besides the input.
The fused variant also performs ``h = x + residual`` in the same pass and returns both
``rmsnorm(h)`` and ``h``, saving a full read + write of the hidden state per layer.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import autotune, warp_configs

MAX_FUSED_N = 65536  # elements; a row must fit in one block


def _flatten(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    return x2


@autotune(warp_configs(), key=["N"])
@triton.jit
def _rmsnorm_fwd_kernel(
    X,
    RES,
    W,
    Y,
    RES_OUT,
    RSTD,
    x_stride,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(RES + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(RES_OUT + row * N + cols, x.to(RES_OUT.dtype.element_ty), mask=mask)
        # Normalise the value as it will be seen by the backward pass (rounded to dtype).
        x = x.to(RES_OUT.dtype.element_ty).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@autotune(warp_configs(), key=["N"])
@triton.jit
def _rmsnorm_bwd_kernel(
    X,
    W,
    DY,
    DRES,
    DX,
    RSTD,
    DW_PART,
    x_stride,
    M,
    N,
    GROUPS,
    BLOCK_N: tl.constexpr,
    HAS_DRES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_N], dtype=tl.float32)
    for row in range(pid, M, GROUPS):
        x = tl.load(X + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + row)
        xhat = x * rstd
        g = dy * w
        c = tl.sum(xhat * g, axis=0) / N
        dx = (g - xhat * c) * rstd
        if HAS_DRES:
            dx += tl.load(DRES + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(DX + row * N + cols, dx.to(DX.dtype.element_ty), mask=mask)
        dw += dy * xhat
    tl.store(DW_PART + pid * N + cols, dw, mask=mask)


def _check(n: int) -> int:
    block_n = triton.next_power_of_2(n)
    if block_n > MAX_FUSED_N:
        raise ValueError(f"hidden size {n} exceeds the single-block limit of {MAX_FUSED_N}")
    return block_n


def rmsnorm_forward(x2, weight, eps, residual2=None):
    m, n = x2.shape
    block_n = _check(n)
    y = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
    rstd = torch.empty(m, dtype=torch.float32, device=x2.device)
    res_out = torch.empty_like(y) if residual2 is not None else y
    _rmsnorm_fwd_kernel[(m,)](
        x2,
        residual2 if residual2 is not None else x2,
        weight,
        y,
        res_out,
        rstd,
        x2.stride(0),
        n,
        eps,
        BLOCK_N=block_n,
        HAS_RES=residual2 is not None,
    )
    return y, rstd, (res_out if residual2 is not None else None)


def rmsnorm_backward(x2, weight, dy2, rstd, dres2=None):
    m, n = x2.shape
    block_n = _check(n)
    groups = max(1, min(m, 128))
    dx = torch.empty((m, n), dtype=x2.dtype, device=x2.device)
    dw_part = torch.empty((groups, n), dtype=torch.float32, device=x2.device)
    _rmsnorm_bwd_kernel[(groups,)](
        x2,
        weight,
        dy2,
        dres2 if dres2 is not None else dy2,
        dx,
        rstd,
        dw_part,
        x2.stride(0),
        m,
        n,
        groups,
        BLOCK_N=block_n,
        HAS_DRES=dres2 is not None,
    )
    return dx, dw_part.sum(0).to(weight.dtype)


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = _flatten(x)
        y, rstd, _ = rmsnorm_forward(x2, weight.contiguous(), eps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        dx, dw = rmsnorm_backward(x2, weight.contiguous(), _flatten(dy.contiguous()), rstd)
        return dx.view(ctx.shape), dw, None


class AddRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        shape = x.shape
        x2, r2 = _flatten(x), _flatten(residual).contiguous()
        y, rstd, h = rmsnorm_forward(x2, weight.contiguous(), eps, residual2=r2)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(h, weight, rstd)
        ctx.shape = shape
        return y.view(shape), h.view(shape)

    @staticmethod
    def backward(ctx, dy, dh):
        h, weight, rstd = ctx.saved_tensors
        n = h.shape[-1]
        dy2 = _flatten(dy.contiguous()) if dy is not None else torch.zeros_like(h)
        dres2 = _flatten(dh.contiguous()) if dh is not None else None
        dx, dw = rmsnorm_backward(h, weight.contiguous(), dy2.view(-1, n), rstd, dres2)
        dx = dx.view(ctx.shape)
        # h = x + residual, so both inputs receive the same gradient.
        return dx, dx, dw, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over the last dimension of ``x``."""
    return RMSNormFn.apply(x, weight, eps)


def add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(rmsnorm(x + residual), x + residual)`` computed in a single pass."""
    return AddRMSNormFn.apply(x, residual, weight, eps)


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        return add_rmsnorm(x, residual, self.weight, self.eps)
