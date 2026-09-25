import pytest
import torch

from fused_kernels.reference import swiglu_ref
from fused_kernels.swiglu import swiglu, swiglu_packed

TOL = {torch.float32: dict(atol=1e-5, rtol=1e-4), torch.float16: dict(atol=2e-2, rtol=2e-2)}
SHAPES = [(4, 64), (3, 5, 96), (2, 1000), (1, 2049)]  # incl. sizes not divisible by BLOCK


def _ab(shape, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = (torch.randn(shape, generator=g) * 2).to(device=device, dtype=dtype)
    b = torch.randn(shape, generator=g).to(device=device, dtype=dtype)
    return a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_reference(shape, dtype, device):
    a, b = _ab(shape, dtype, device)
    torch.testing.assert_close(swiglu(a, b), swiglu_ref(a, b), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_autograd(shape, dtype, device):
    a, b = _ab(shape, dtype, device)
    dy = torch.randn_like(a)
    a1, b1 = a.clone().requires_grad_(), b.clone().requires_grad_()
    swiglu(a1, b1).backward(dy)
    a2, b2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    swiglu_ref(a2.float(), b2.float()).to(dtype).backward(dy)
    torch.testing.assert_close(a1.grad, a2.grad, **TOL[dtype])
    torch.testing.assert_close(b1.grad, b2.grad, **TOL[dtype])


@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 100), (1, 513)])
def test_packed_matches_unpacked(shape, device):
    a, b = _ab(shape, torch.float32, device)
    x = torch.cat([a, b], dim=-1)
    torch.testing.assert_close(swiglu_packed(x), swiglu_ref(a, b), atol=1e-5, rtol=1e-4)

    dy = torch.randn_like(a)
    xa = x.clone().requires_grad_()
    swiglu_packed(xa).backward(dy)
    a2, b2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    swiglu_ref(a2, b2).backward(dy)
    expected = torch.cat([a2.grad, b2.grad], dim=-1)
    torch.testing.assert_close(xa.grad, expected, atol=1e-5, rtol=1e-4)


def test_non_contiguous_gate_up(device):
    x = torch.randn(6, 128, device=device)
    a, b = x[:, :64], x[:, 64:]  # row stride 128, not 64
    torch.testing.assert_close(swiglu(a, b), swiglu_ref(a, b), atol=1e-5, rtol=1e-4)


def test_extreme_gate_values_are_finite(device):
    a = torch.tensor([[-80.0, -20.0, 0.0, 20.0, 80.0]], device=device)
    b = torch.ones_like(a)
    y = swiglu(a, b)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, swiglu_ref(a, b), atol=1e-5, rtol=1e-4)


def test_shape_validation(device):
    with pytest.raises(ValueError):
        swiglu(torch.zeros(2, 4, device=device), torch.zeros(2, 5, device=device))
    with pytest.raises(ValueError):
        swiglu_packed(torch.zeros(2, 5, device=device))
