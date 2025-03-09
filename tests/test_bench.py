import os
import subprocess
import sys

import pytest
import torch

from fused_kernels import bench


def test_ideal_bytes_accounting():
    t, n, e = 10, 100, 2
    assert bench.ideal_bytes("rmsnorm", "fwd", t, hidden=n, elem=e) == 2 * t * n * e
    assert bench.ideal_bytes("rmsnorm", "bwd", t, hidden=n, elem=e) == 3 * t * n * e
    assert bench.ideal_bytes("add_rmsnorm", "fwd", t, hidden=n, elem=e) == 4 * t * n * e
    assert bench.ideal_bytes("swiglu", "fwd", t, inter=n, elem=e) == 3 * t * n * e
    assert bench.ideal_bytes("swiglu", "bwd", t, inter=n, elem=e) == 5 * t * n * e
    # rope reads and writes q and k once each
    assert (
        bench.ideal_bytes("rope", "fwd", t, hq=8, hk=2, head_dim=64, elem=e) == 2 * t * 10 * 64 * e
    )


def test_ideal_bytes_rejects_unknown():
    with pytest.raises(ValueError):
        bench.ideal_bytes("nope", "fwd", 1)
    with pytest.raises(ValueError):
        bench.ideal_bytes("rmsnorm", "sideways", 1, hidden=1)


def test_smoke_run_covers_every_op_and_direction(capsys):
    bench.main(["--smoke"])
    out = capsys.readouterr().out
    for op in bench.OPS:
        for direction in ("fwd", "bwd"):
            assert f"| {op} | {direction} |" in out
    assert "| triton |" in out and "| eager |" in out


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU default")
def test_cli_smoke_works_in_a_clean_environment_without_a_gpu():
    """Regression: importing the package must enable the interpreter *before* triton loads."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRITON_")}
    proc = subprocess.run(
        [sys.executable, "-m", "fused_kernels.bench", "--smoke", "--ops", "swiglu"],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "SMOKE MODE" in proc.stdout
