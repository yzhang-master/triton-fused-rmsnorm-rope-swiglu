"""Helpers shared by all kernels."""

from __future__ import annotations

import os

import triton


def interpreting() -> bool:
    """True when Triton runs kernels on the CPU interpreter (no GPU available)."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def autotune(configs: list[triton.Config], key: list[str], **kwargs):
    """``triton.autotune`` that degrades to the first config under the CPU interpreter.

    The interpreter executes kernels in Python, so benchmarking every config would only
    slow the test-suite down; the first config is always a valid launch.
    """
    if interpreting() or os.environ.get("FUSED_KERNELS_NO_AUTOTUNE") == "1":
        configs = configs[:1]
    return triton.autotune(configs=configs, key=key, **kwargs)


def warp_configs(warps: tuple[int, ...] = (2, 4, 8, 16)) -> list[triton.Config]:
    return [triton.Config({}, num_warps=w) for w in warps]
