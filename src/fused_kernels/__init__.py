"""Fused Triton kernels for the memory-bound ops of a Llama-style transformer block.

Without a CUDA device Triton has no backend, so importing this package on such a machine
turns on Triton's CPU interpreter (``TRITON_INTERPRET=1``). It must be decided before
``triton`` is first imported, hence it happens here. Set the variable yourself to override.
"""

import os

import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from .reference import add_rmsnorm_ref, rmsnorm_ref, rope_ref, swiglu_ref
from .rmsnorm import FusedRMSNorm, add_rmsnorm, rmsnorm
from .rope import apply_rope, build_rope_cache
from .swiglu import swiglu, swiglu_packed

__version__ = "0.1.0"

__all__ = [
    "FusedRMSNorm",
    "add_rmsnorm",
    "add_rmsnorm_ref",
    "apply_rope",
    "build_rope_cache",
    "rmsnorm",
    "rmsnorm_ref",
    "rope_ref",
    "swiglu",
    "swiglu_packed",
    "swiglu_ref",
]
