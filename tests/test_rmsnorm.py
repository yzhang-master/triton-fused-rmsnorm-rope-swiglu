import pytest
import torch

from fused_kernels.reference import add_rmsnorm_ref, rmsnorm_ref
from fused_kernels.rmsnorm import FusedRMSNorm, add_rmsnorm, rmsnorm

TOL = {torch.float32: dict(atol=1e-5, rtol=1e-4), torch.float16: dict(atol=2e-2, rtol=2e-2)}
SHAPES = [(4, 64), (3, 5, 96), (7, 130), (2, 1000)]  # incl. non-power-of-two hidden sizes


def _data(shape, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g).to(device=device, dtype=dtype)
    w = (torch.randn(shape[-1], generator=g) * 0.5 + 1).to(device=device, dtype=dtype)
    return x, w


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_reference(shape, dtype, device):
    x, w = _data(shape, dtype, device)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_ref(x, w), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_autograd(shape, dtype, device):
    x, w = _data(shape, dtype, device)
    dy = torch.randn_like(x)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    xb, wb = x.clone().requires_grad_(), w.clone().requires_grad_()
    rmsnorm(xa, wa).backward(dy)
    rmsnorm_ref(xb.float(), wb.float()).to(dtype).backward(dy)
    tol = TOL[dtype] if dtype == torch.float32 else dict(atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(xa.grad, xb.grad, **tol)
    torch.testing.assert_close(wa.grad, wb.grad, **tol)


def test_non_contiguous_input(device):
    x, w = _data((6, 64), torch.float32, device)
    xt = torch.randn(64, 6, device=device).t()  # stride (1, 6): last dim not contiguous
    torch.testing.assert_close(rmsnorm(xt, w), rmsnorm_ref(xt, w), atol=1e-5, rtol=1e-4)
    xs = torch.randn(6, 128, device=device)[:, ::2]  # strided view
    torch.testing.assert_close(rmsnorm(xs, w), rmsnorm_ref(xs, w), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 100)])
def test_fused_add_forward(shape, device):
    x, w = _data(shape, torch.float32, device)
    r = torch.randn_like(x)
    y, h = add_rmsnorm(x, r, w)
    y_ref, h_ref = add_rmsnorm_ref(x, r, w)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(h, h_ref)


def test_fused_add_backward_uses_both_outputs(device):
    """Loss depends on both the normalised output and the new residual stream."""
    x, w = _data((5, 96), torch.float32, device)
    r = torch.randn_like(x)
    dy, dh = torch.randn_like(x), torch.randn_like(x)

    xa, ra, wa = (t.clone().requires_grad_() for t in (x, r, w))
    y, h = add_rmsnorm(xa, ra, wa)
    ((y * dy).sum() + (h * dh).sum()).backward()

    xb, rb, wb = (t.clone().requires_grad_() for t in (x, r, w))
    yb, hb = add_rmsnorm_ref(xb, rb, wb)
    ((yb * dy).sum() + (hb * dh).sum()).backward()

    for a, b in ((xa.grad, xb.grad), (ra.grad, rb.grad), (wa.grad, wb.grad)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_fused_add_unused_residual_output(device):
    x, w = _data((3, 64), torch.float32, device)
    r = torch.randn_like(x)
    xa, ra, wa = (t.clone().requires_grad_() for t in (x, r, w))
    y, _ = add_rmsnorm(xa, ra, wa)
    y.sum().backward()
    xb, rb, wb = (t.clone().requires_grad_() for t in (x, r, w))
    add_rmsnorm_ref(xb, rb, wb)[0].sum().backward()
    torch.testing.assert_close(xa.grad, xb.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(wa.grad, wb.grad, atol=1e-4, rtol=1e-4)


def test_module_matches_functional(device):
    x, _ = _data((4, 64), torch.float32, device)
    m = FusedRMSNorm(64).to(device)
    torch.testing.assert_close(m(x), rmsnorm_ref(x, m.weight))


def test_rejects_oversized_hidden(device):
    x = torch.empty(1, 70000, device=device)
    with pytest.raises(ValueError):
        rmsnorm(x, torch.ones(70000, device=device))
