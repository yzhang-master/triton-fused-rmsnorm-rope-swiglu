"""Run on a real GPU when one exists, otherwise fall back to Triton's CPU interpreter.

``TRITON_INTERPRET`` must be set before ``triton`` is imported, so it is decided here,
at collection time, before any test module touches the kernels.
"""

import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ["TRITON_INTERPRET"] = "1"
    os.environ.setdefault("FUSED_KERNELS_NO_AUTOTUNE", "1")


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if HAS_CUDA else "cpu"
