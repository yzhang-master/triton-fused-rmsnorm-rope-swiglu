"""SwiGLU activation ``silu(gate) * up`` (forward and backward) in Triton.

    y  = silu(a) * b,                 sig = sigmoid(a)
    da = dy * b * sig * (1 + a * (1 - sig))
    db = dy * a * sig

Unfused PyTorch materialises ``silu(a)`` and then multiplies, i.e. two extra HBM round
trips of the [tokens, intermediate] tensor and an extra saved activation. The fused
kernel reads ``a`` and ``b`` once and writes ``y`` once; backward recomputes ``sig`` from
``a`` instead of saving it. ``swiglu_packed`` takes the output of a single fused
gate+up projection ([..., 2*I], gate first) and writes the packed gradient directly,
avoiding the ``chunk`` / ``cat`` copies autograd would otherwise add.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._common import autotune


def _configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK": blk}, num_warps=w) for blk in (256, 512, 1024, 2048) for w in (4, 8)
    ]


@autotune(_configs(), key=["INTER"])
@triton.jit
def _swiglu_fwd_kernel(A, B, Y, a_stride, b_stride, y_stride, INTER, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < INTER
    a = tl.load(A + row * a_stride + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + row * b_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * b
    tl.store(Y + row * y_stride + cols, y.to(Y.dtype.element_ty), mask=mask)


@autotune(_configs(), key=["INTER"])
@triton.jit
def _swiglu_bwd_kernel(
    A, B, DY, DA, DB, a_stride, b_stride, dy_stride, da_stride, db_stride, INTER,
    BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < INTER
    a = tl.load(A + row * a_stride + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + row * b_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * dy_stride + cols, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(a)
    da = dy * b * sig * (1.0 + a * (1.0 - sig))
    db = dy * a * sig
    tl.store(DA + row * da_stride + cols, da.to(DA.dtype.element_ty), mask=mask)
    tl.store(DB + row * db_stride + cols, db.to(DB.dtype.element_ty), mask=mask)


def _rows(x: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    return x2 if x2.stride(-1) == 1 else x2.contiguous()


def _grid(m, i):
    return lambda meta: (m, triton.cdiv(i, meta["BLOCK"]))


def _fwd(a2, b2):
    m, i = a2.shape
    y = torch.empty((m, i), dtype=a2.dtype, device=a2.device)
    _swiglu_fwd_kernel[_grid(m, i)](a2, b2, y, a2.stride(0), b2.stride(0), y.stride(0), i)
    return y


def _bwd(a2, b2, dy2, da, db):
    m, i = a2.shape
    _swiglu_bwd_kernel[_grid(m, i)](
        a2, b2, dy2, da, db,
        a2.stride(0), b2.stride(0), dy2.stride(0), da.stride(0), db.stride(0),
        i,
    )  # fmt: skip


class SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        shape = gate.shape
        a2, b2 = _rows(gate), _rows(up)
        ctx.save_for_backward(a2, b2)
        ctx.shape = shape
        return _fwd(a2, b2).view(shape)

    @staticmethod
    def backward(ctx, dy):
        a2, b2 = ctx.saved_tensors
        da = torch.empty_like(a2)
        db = torch.empty_like(b2)
        _bwd(a2, b2, _rows(dy), da, db)
        return da.view(ctx.shape), db.view(ctx.shape)


class SwiGLUPackedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        *lead, two_i = x.shape
        if two_i % 2:
            raise ValueError("last dimension must be even (gate and up concatenated)")
        i = two_i // 2
        x2 = _rows(x)
        ctx.save_for_backward(x2)
        ctx.lead = lead
        return _fwd(x2[:, :i], x2[:, i:]).view(*lead, i)

    @staticmethod
    def backward(ctx, dy):
        (x2,) = ctx.saved_tensors
        i = x2.shape[-1] // 2
        dx = torch.empty_like(x2)
        _bwd(x2[:, :i], x2[:, i:], _rows(dy), dx[:, :i], dx[:, i:])
        return dx.view(*ctx.lead, 2 * i)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for tensors of identical shape."""
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    return SwiGLUFn.apply(gate, up)


def swiglu_packed(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU on a packed ``[..., 2*INTER]`` tensor whose first half is the gate."""
    return SwiGLUPackedFn.apply(x)
