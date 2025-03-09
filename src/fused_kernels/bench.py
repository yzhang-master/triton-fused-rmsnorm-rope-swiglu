"""Benchmark harness: Triton kernels vs eager PyTorch vs ``torch.compile``.

Every implementation of an op is verified against the eager reference before it is timed,
and all three are credited with the same *ideal* HBM traffic (see ``ideal_bytes``), so the
reported GB/s is an effective bandwidth that is directly comparable across them: an
implementation that moves more bytes than necessary simply looks slower.

    python -m fused_kernels.bench                     # needs a CUDA GPU
    python -m fused_kernels.bench --peak-gbps 3350    # also report % of peak HBM bandwidth
    python -m fused_kernels.bench --smoke             # CPU interpreter, tiny shapes, CI only

``--smoke`` exists to exercise every code path in CI; its timings are meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .reference import add_rmsnorm_ref, rmsnorm_ref, rope_ref, swiglu_ref

OPS = ("rmsnorm", "add_rmsnorm", "rope", "swiglu")


def ideal_bytes(
    op: str,
    direction: str,
    tokens: int,
    hidden: int = 0,
    hq: int = 0,
    hk: int = 0,
    head_dim: int = 0,
    inter: int = 0,
    elem: int = 2,
) -> int:
    """Minimum HBM traffic in bytes (per-row scalars and weights are negligible, ignored).

    =============  ===========================  ==================================
    op             forward                      backward
    =============  ===========================  ==================================
    rmsnorm        read x, write y              read x, dy; write dx
    add_rmsnorm    read x, res; write y, h      read h, dy, dh; write dx
    rope           read + write q, k            read + write dq, dk
    swiglu         read a, b; write y           read a, b, dy; write da, db
    =============  ===========================  ==================================
    """
    if direction not in ("fwd", "bwd"):
        raise ValueError(direction)
    if op == "rmsnorm":
        t = tokens * hidden * elem
        return (2 if direction == "fwd" else 3) * t
    if op == "add_rmsnorm":
        return 4 * tokens * hidden * elem
    if op == "rope":
        return 2 * tokens * (hq + hk) * head_dim * elem
    if op == "swiglu":
        t = tokens * inter * elem
        return (3 if direction == "fwd" else 5) * t
    raise ValueError(op)


@dataclass
class Row:
    op: str
    direction: str
    tokens: int
    impl: str
    ms: float
    gbps: float
    pct_peak: float | None
    speedup_vs_eager: float | None = None
    speedup_vs_compile: float | None = None


def _as_tuple(out):
    return out if isinstance(out, tuple) else (out,)


def _fwd(f, inputs):
    return lambda: f(*inputs)


def _bwd(f, inputs, grads):
    outs = _as_tuple(f(*inputs))

    def run():
        torch.autograd.grad(outs, inputs, grads, retain_graph=True)

    return run


def _build(op, args, device, dtype):
    """Return (impls, inputs, grad_outputs, byte kwargs) for one op."""
    from .rmsnorm import add_rmsnorm, rmsnorm
    from .rope import apply_rope, build_rope_cache
    from .swiglu import swiglu

    t, n = args.tokens, args.hidden

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

    if op == "rmsnorm":
        x, w = rand(t, n), rand(n)
        impls = {"eager": rmsnorm_ref, "triton": rmsnorm}
        return impls, (x, w), (torch.randn_like(x),), dict(hidden=n)
    if op == "add_rmsnorm":
        x, r, w = rand(t, n), rand(t, n), rand(n)
        impls = {"eager": add_rmsnorm_ref, "triton": add_rmsnorm}
        return impls, (x, r, w), (torch.randn_like(x), torch.randn_like(x)), dict(hidden=n)
    if op == "rope":
        hq, hk, d = args.q_heads, args.kv_heads, args.head_dim
        q, k = rand(1, t, hq, d), rand(1, t, hk, d)
        cos, sin = build_rope_cache(t, d, device=device)
        impls = {
            "eager": lambda q, k: (rope_ref(q, cos, sin), rope_ref(k, cos, sin)),
            "triton": lambda q, k: apply_rope(q, k, cos, sin),
        }
        return (
            impls,
            (q, k),
            (torch.randn_like(q), torch.randn_like(k)),
            dict(hq=hq, hk=hk, head_dim=d),
        )
    if op == "swiglu":
        a, b = rand(t, args.inter), rand(t, args.inter)
        impls = {"eager": swiglu_ref, "triton": swiglu}
        return impls, (a, b), (torch.randn_like(a),), dict(inter=args.inter)
    raise ValueError(op)


def _time_ms(fn: Callable[[], object], smoke: bool) -> float:
    if smoke:
        fn()
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1e3
    import triton

    return float(triton.testing.do_bench(fn, warmup=50, rep=200, return_mode="median"))


def _verify(impls, inputs, dtype):
    tol = dict(atol=5e-2, rtol=5e-2) if dtype != torch.float32 else dict(atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        ref = _as_tuple(impls["eager"](*inputs))
        for name, f in impls.items():
            if name == "eager":
                continue
            for got, want in zip(_as_tuple(f(*inputs)), ref, strict=True):
                torch.testing.assert_close(got, want, **tol, msg=lambda m, n=name: f"{n}: {m}")


def run(args) -> list[Row]:
    device = "cpu" if args.smoke else "cuda"
    dtype = torch.float32 if args.smoke else getattr(torch, args.dtype)
    elem = torch.finfo(dtype).bits // 8
    rows: list[Row] = []
    for op in args.ops:
        impls, inputs, grads, kw = _build(op, args, device, dtype)
        _verify(impls, inputs, dtype)
        if not args.smoke:
            impls["compile"] = torch.compile(impls["eager"])
        for direction in ("fwd", "bwd"):
            nbytes = ideal_bytes(op, direction, args.tokens, elem=elem, **kw)
            timed: dict[str, float] = {}
            for name, f in impls.items():
                if name == "compile":
                    torch._dynamo.reset()  # keep each shape under the recompile limit
                fn = _fwd(f, inputs) if direction == "fwd" else _bwd(f, inputs, grads)
                timed[name] = _time_ms(fn, args.smoke)
            for name, ms in timed.items():
                gbps = nbytes / (ms * 1e-3) / 1e9
                rows.append(
                    Row(
                        op=op,
                        direction=direction,
                        tokens=args.tokens,
                        impl=name,
                        ms=ms,
                        gbps=gbps,
                        pct_peak=(100 * gbps / args.peak_gbps) if args.peak_gbps else None,
                        speedup_vs_eager=timed["eager"] / ms,
                        speedup_vs_compile=(timed["compile"] / ms) if "compile" in timed else None,
                    )
                )
    return rows


def to_markdown(rows: list[Row]) -> str:
    head = "| op | pass | tokens | impl | ms | GB/s | % peak | vs eager | vs compile |\n"
    head += "|---|---|---:|---|---:|---:|---:|---:|---:|\n"

    def fmt(v, spec, suffix=""):
        return "-" if v is None else format(v, spec) + suffix

    lines = [
        f"| {r.op} | {r.direction} | {r.tokens} | {r.impl} | {r.ms:.4f} | {r.gbps:.0f} | "
        f"{fmt(r.pct_peak, '.0f')} | {fmt(r.speedup_vs_eager, '.2f', 'x')} | "
        f"{fmt(r.speedup_vs_compile, '.2f', 'x')} |"
        for r in rows
    ]
    return head + "\n".join(lines) + "\n"


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--inter", type=int, default=14336, help="MLP intermediate size")
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--peak-gbps", type=float, default=None, help="device peak HBM GB/s")
    p.add_argument("--out", default="results", help="directory for the .json / .md output")
    p.add_argument("--smoke", action="store_true", help="CPU interpreter, tiny shapes (CI only)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    if args.smoke:
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise SystemExit("--smoke needs the CPU interpreter: run with TRITON_INTERPRET=1")
        args.hidden, args.inter, args.q_heads, args.kv_heads, args.head_dim = 96, 80, 4, 2, 32
        args.tokens = [16]
    elif not torch.cuda.is_available():
        raise SystemExit("No CUDA device found. Use --smoke to exercise the harness on CPU.")

    all_rows: list[Row] = []
    for tokens in args.tokens:
        sub = argparse.Namespace(**{**vars(args), "tokens": tokens})
        all_rows += run(sub)

    md = to_markdown(all_rows)
    if args.smoke:
        print("SMOKE MODE: CPU interpreter timings are NOT performance data.\n")
        print(md)
        return
    print(md)
    meta = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": args.dtype,
        "hidden": args.hidden,
        "inter": args.inter,
        "heads": [args.q_heads, args.kv_heads, args.head_dim],
    }
    import triton

    meta["triton"] = triton.__version__
    os.makedirs(args.out, exist_ok=True)
    stem = meta["gpu"].lower().replace(" ", "-")
    with open(os.path.join(args.out, f"{stem}.json"), "w") as f:
        json.dump({"meta": meta, "rows": [asdict(r) for r in all_rows]}, f, indent=2)
    with open(os.path.join(args.out, f"{stem}.md"), "w") as f:
        f.write(f"GPU: {meta['gpu']}  \n")
        f.write(f"torch {meta['torch']} / triton {meta['triton']} / dtype {args.dtype}\n\n")
        f.write(md)


if __name__ == "__main__":
    main()
