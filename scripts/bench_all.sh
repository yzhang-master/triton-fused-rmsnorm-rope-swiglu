#!/usr/bin/env bash
# Run the full kernel benchmark on the local GPU and write results/<gpu>.{json,md}.
# Usage: scripts/bench_all.sh [PEAK_HBM_GBPS]   e.g. scripts/bench_all.sh 3350
set -euo pipefail
cd "$(dirname "$0")/.."

PEAK_ARGS=()
if [[ $# -ge 1 ]]; then PEAK_ARGS=(--peak-gbps "$1"); fi

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'"
python -m pytest -q                      # kernels must be correct on this GPU before we time them
python -m fused_kernels.bench "${PEAK_ARGS[@]}"
