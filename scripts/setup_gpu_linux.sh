#!/usr/bin/env bash
# Finish GPU setup on a Linux box (e.g. RTX 4090 / sm_89 Ada).
#
# requirements.txt already declares everything (incl. `tensorflow[and-cuda]` on Linux and
# `numpy<2`). This script handles the one thing pip can't express: the venv ends up with
# BOTH cu12 and cu13 NVIDIA wheels, and TensorFlow 2.21 needs the cu12 `libcusolver.so.11`
# preloaded or it silently runs on CPU. It installs a `.pth` so that preload happens for
# every interpreter in the venv (uv run, activated shell, Jupyter, ad-hoc scripts), then
# verifies torch + TF both see the GPU.
#
# Run from the repo root, after `uv venv` + `uv pip install -r requirements.txt`:
#     bash scripts/setup_gpu_linux.sh
#
# macOS/non-Linux: not needed — skip this entirely.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Not Linux — nothing to do (the GPU TF fix is Linux-only)."
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
    echo "ERROR: no .venv found. Run 'uv venv --python 3.11 && uv pip install -r requirements.txt' first." >&2
    exit 1
fi

SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
echo "site-packages: $SITE"

# Install a .pth that preloads cu12 libcusolver.so.11 at interpreter startup, for any
# entry point that does NOT import the `src` package first (notebooks, scripts).
# `zzz_` prefix so it sorts after the nvidia wheels' own .pth files.
PTH="$SITE/zzz_mmml_cuda_preload.pth"
cat > "$PTH" <<'PTHEOF'
import os, sys, glob, ctypes
if sys.platform == 'linux':
    for _d in sys.path:
        for _so in glob.glob(os.path.join(_d, 'nvidia', 'cusolver', 'lib', 'libcusolver.so.11')):
            try: ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL); break
            except OSError: pass
PTHEOF
echo "installed: $PTH"

echo ""
echo "=== Verifying GPU ==="
uv run python - <<'PYEOF'
import torch
print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("TF:", tf.__version__, "GPUs:", gpus)
assert torch.cuda.is_available(), "torch does not see the GPU"
assert gpus, "TF does not see the GPU — libcusolver.so.11 preload likely failed"
print("\nOK: both torch and TF see the GPU.")
PYEOF
