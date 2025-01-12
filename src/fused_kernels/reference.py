"""Plain PyTorch references. Everything is computed in fp32 and cast once at the end."""

from __future__ import annotations

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    h = x + residual
    return rmsnorm_ref(h, weight, eps), h


def rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Half-split ("rotate_half") RoPE on ``x`` of shape [B, S, H, D]."""
    _, s, _, d = x.shape
    if positions is None:
        c, sn = cos[:s][None, :, None, :], sin[:s][None, :, None, :]
    else:
        c, sn = cos[positions][:, :, None, :], sin[positions][:, :, None, :]
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2 :]
    out = torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], dim=-1)
    return out.to(x.dtype)


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    g = gate.float()
    return (g * torch.sigmoid(g) * up.float()).to(gate.dtype)
