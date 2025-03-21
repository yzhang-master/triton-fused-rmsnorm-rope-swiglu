# triton-fused-rmsnorm-rope-swiglu

> **Era:** 2025+ · Staff AI Engineer focus area (fused-kernel optimization for LLM training and serving). Written in 2026 on Triton 3.x and PyTorch 2.x.

Fused [Triton](https://github.com/triton-lang/triton) kernels — **forward and backward** — for
the memory-bound ops that sit between the matmuls of a Llama-style transformer block:

| Op | Fusion | API |
|---|---|---|
| RMSNorm | one pass per row, saves only `rstd` | `rmsnorm(x, w)` |
| residual-add + RMSNorm | add and norm in one pass, returns both outputs | `add_rmsnorm(x, residual, w)` |
| RoPE | q **and** k in one launch, GQA, position ids, in-place | `apply_rope(q, k, cos, sin)` |
| SwiGLU | `silu(gate) * up`, sigmoid recomputed in backward | `swiglu(gate, up)` / `swiglu_packed(x)` |

All of them are `torch.autograd.Function`s, compute in fp32 and round once on store.

## Problem

Decoder blocks spend a large share of their non-matmul time moving activations through
HBM. These ops do almost no arithmetic per byte, so their runtime is set by how many times
the activation tensor is read and written. Eager PyTorch launches one kernel per primitive
(`pow`, `mean`, `rsqrt`, `mul`, ... plus fp32 up-/down-casts), each paying a full round trip.

Minimum HBM traffic, in units of one `[tokens, hidden]` tensor (analytical, not measured):

| Op | Unfused, at best | Fused (this repo) |
|---|---|---|
| `x + res`, then RMSNorm (fwd) | 5 (add: 3, norm: 2) | **4** |
| `silu(a) * b` (fwd) | 5 (silu: 2, mul: 3) | **3** |
| SwiGLU saved activations | `a`, `b`, `silu(a)` | `a`, `b` |

and "at best" is generous: eager RMSNorm with an fp32 upcast is several kernels.

## Method

**RMSNorm.** `y = x · rstd · w`, `rstd = 1/√(mean(x²)+ε)`. With `x̂ = x·rstd`, `g = dy·w`:

```
dx = rstd · (g − x̂ · mean(g · x̂))        dw = Σ_rows dy · x̂
```

One program per row. `dw` is reduced through a `[groups, N]` fp32 partial buffer summed on
the host instead of atomics, so the result is **deterministic**. The fused variant computes
`h = x + residual` in the same pass; backward adds the incoming gradient of `h` to `dx`, and
both inputs receive that same gradient.

**RoPE** (half-split layout). With `(x1, x2) = (x[i], x[i + D/2])`:
`out1 = x1·cos − x2·sin`, `out2 = x2·cos + x1·sin`. It is a rotation, so backward is the same
kernel with `sin` negated and nothing needs saving. One program handles one token and every
query *and* key head. `inplace=True` is safe because each element pair is fully loaded before
the same program stores it, and the kernel is autotuned with `restore_value` so tuning trials
do not rotate the data repeatedly.

**SwiGLU.** `y = silu(a)·b`; `da = dy·b·σ(a)·(1 + a(1−σ(a)))`, `db = dy·a·σ(a)`. The packed
variant reads a fused `[..., 2I]` gate+up projection through strided views and writes the packed
gradient directly, avoiding the `chunk`/`cat` copies autograd would add.

**Autotuning.** `num_warps` for the row kernels, `BLOCK × num_warps` for SwiGLU, keyed on the
size of the reduced dimension. Under the CPU interpreter it falls back to the first config.

## Correctness

`pytest` runs **68 tests** on any machine: on CPU through Triton's interpreter, on a GPU natively.
They compare forward *and* backward against fp32 PyTorch autograd (fp32 and fp16), including
non-power-of-two sizes, strided/non-contiguous inputs, GQA, explicit positions, in-place RoPE,
saturating SwiGLU inputs and input validation.

The tests were **mutation-checked**: deliberately breaking each backward formula, the RoPE
inverse rotation and the in-place path makes them fail. A bug found along the way (the CLI
loading Triton before the interpreter flag was set) has a regression test.

## Results

**Not yet measured.** The development machine had no GPU, and I will not publish numbers I did
not measure. The harness is complete and unit-tested; on a CUDA machine:

```bash
pip install -e ".[dev]"
scripts/bench_all.sh 3350          # optional arg: your GPU's peak HBM GB/s, for "% of peak"
```

It first runs the test-suite on that GPU, then verifies every implementation against the eager
reference, then times Triton vs eager vs `torch.compile` for each op and direction at 2k/8k/32k
tokens (Llama-3-8B shapes by default), and writes `results/<gpu>.json` and `results/<gpu>.md`
with the library versions used.

GB/s is *effective* bandwidth: every implementation is credited with the same ideal traffic
(`fused_kernels.bench.ideal_bytes`), so an implementation that moves extra bytes shows up as
slower rather than being flattered by its own traffic.

| GPU | op | pass | Triton GB/s | eager GB/s | compile GB/s |
|---|---|---|---|---|---|
| _pending_ | | | | | |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[dev]"
pytest                               # CPU interpreter, or native on a GPU
python -m fused_kernels.bench --smoke  # CPU: exercise the harness (timings meaningless)
```

## Limitations

- RMSNorm hidden size is limited to 65 536 (a row must fit in one block).
- Index arithmetic is int32: tensors of 2³¹ or more elements are unsupported.
- RoPE implements the half-split (`rotate_half`) layout, not interleaved pairs.
- The interpreter is for correctness only and is orders of magnitude slower than a GPU.

## References

- B. Zhang, R. Sennrich. *Root Mean Square Layer Normalization.* 2019. arXiv:1910.07467
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. arXiv:2104.09864
- N. Shazeer. *GLU Variants Improve Transformer.* 2020. arXiv:2002.05202
- P. Tillet, H. T. Kung, D. Cox. *Triton: an intermediate language and compiler for tiled neural network computations.* MAPL 2019
- P.-L. Hsu et al. *Liger Kernel: Efficient Triton Kernels for LLM Training.* 2024. arXiv:2410.10989 (closely related prior work)

## License

MIT
