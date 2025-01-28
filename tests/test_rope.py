import pytest
import torch

from fused_kernels.reference import rope_ref
from fused_kernels.rope import apply_rope, build_rope_cache

TOL = {torch.float32: dict(atol=1e-5, rtol=1e-5), torch.float16: dict(atol=5e-3, rtol=5e-3)}
# (batch, seq, q_heads, kv_heads, head_dim): MHA, GQA, non-power-of-two heads / head_dim
CONFIGS = [(2, 5, 4, 4, 64), (1, 7, 8, 2, 64), (2, 3, 6, 3, 96), (1, 4, 5, 1, 32)]


def _qk(b, s, hq, hk, d, dtype, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, s, hq, d, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(b, s, hk, d, generator=g).to(device=device, dtype=dtype)
    return q, k


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_matches_reference(cfg, dtype, device):
    b, s, hq, hk, d = cfg
    q, k = _qk(*cfg, dtype, device)
    cos, sin = build_rope_cache(32, d, device=device)
    qo, ko = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(qo, rope_ref(q, cos, sin), **TOL[dtype])
    torch.testing.assert_close(ko, rope_ref(k, cos, sin), **TOL[dtype])


def test_explicit_positions(device):
    b, s, hq, hk, d = 2, 6, 4, 2, 64
    q, k = _qk(b, s, hq, hk, d, torch.float32, device)
    cos, sin = build_rope_cache(64, d, device=device)
    pos = torch.randint(0, 64, (b, s), device=device)
    qo, ko = apply_rope(q, k, cos, sin, positions=pos)
    torch.testing.assert_close(qo, rope_ref(q, cos, sin, pos), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ko, rope_ref(k, cos, sin, pos), atol=1e-5, rtol=1e-5)


def test_inplace_overwrites_inputs(device):
    q, k = _qk(2, 5, 8, 2, 64, torch.float32, device)
    cos, sin = build_rope_cache(16, 64, device=device)
    q_exp, k_exp = rope_ref(q, cos, sin), rope_ref(k, cos, sin)
    qo, ko = apply_rope(q, k, cos, sin, inplace=True)
    assert qo.data_ptr() == q.data_ptr() and ko.data_ptr() == k.data_ptr()
    torch.testing.assert_close(q, q_exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k, k_exp, atol=1e-5, rtol=1e-5)


def test_rotation_preserves_norm(device):
    q, k = _qk(1, 9, 4, 4, 64, torch.float32, device)
    cos, sin = build_rope_cache(16, 64, device=device)
    qo, _ = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(qo.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("cfg", CONFIGS[:3])
def test_backward_matches_autograd(cfg, inplace, device):
    b, s, hq, hk, d = cfg
    q, k = _qk(*cfg, torch.float32, device)
    cos, sin = build_rope_cache(16, d, device=device)
    dq, dk = torch.randn_like(q), torch.randn_like(k)

    q1, k1 = q.clone().requires_grad_(), k.clone().requires_grad_()
    # in-place ops on a leaf are illegal, so rotate a non-leaf copy
    qo, ko = apply_rope(q1 * 1.0, k1 * 1.0, cos, sin, inplace=inplace)
    ((qo * dq).sum() + (ko * dk).sum()).backward()

    q2, k2 = q.clone().requires_grad_(), k.clone().requires_grad_()
    ((rope_ref(q2, cos, sin) * dq).sum() + (rope_ref(k2, cos, sin) * dk).sum()).backward()
    torch.testing.assert_close(q1.grad, q2.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k1.grad, k2.grad, atol=1e-5, rtol=1e-5)


def test_non_contiguous_heads_dim(device):
    """q sliced out of a fused qkv projection has a non-trivial head stride."""
    b, s, h, d = 2, 4, 4, 64
    qkv = torch.randn(b, s, 3 * h, d, device=device)
    q, k = qkv[:, :, :h], qkv[:, :, h : 2 * h]
    cos, sin = build_rope_cache(8, d, device=device)
    qo, ko = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(qo, rope_ref(q, cos, sin), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ko, rope_ref(k, cos, sin), atol=1e-5, rtol=1e-5)


def test_input_validation(device):
    q, k = _qk(1, 4, 2, 2, 64, torch.float32, device)
    cos, sin = build_rope_cache(8, 64, device=device)
    with pytest.raises(ValueError):
        apply_rope(q[..., :63], k, cos, sin)
    with pytest.raises(ValueError):
        apply_rope(q, k, cos[:, :16], sin[:, :16])
