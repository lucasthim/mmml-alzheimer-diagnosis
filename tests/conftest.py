"""Shared pytest fixtures/config for the deepbrain skull-stripping tests.

Puts ``src/data_preprocessing`` on ``sys.path`` so ``deepbrain_skull_strip`` (whose
imports are all absolute) can be imported directly, and silences TensorFlow's startup
logging so test output stays readable.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # errors only; quiets TF startup noise

# Linux GPU fix: preload cu12 libcusolver.so.11 before TF imports, else TF 2.21 can't
# find it and silently falls back to CPU. No-op off Linux. See src/_cuda_preload.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from _cuda_preload import preload_cusolver  # noqa: E402

    preload_cusolver()
except Exception:
    pass

# TensorFlow MUST be imported before ants (ITK). Both ship an OpenMP runtime, and on
# macOS, if ITK's loads first, TF's session.run deadlocks. Importing TF here in conftest
# guarantees it initializes before any test module does `import ants`.
import tensorflow  # noqa: F401,E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data_preprocessing"))


@pytest.fixture
def dummy_volume():
    """A synthetic 3D MRI-like volume: dark background with a bright central sphere.

    Must be non-zero — ``Extractor.run`` divides by ``np.max(img)``, so an all-zeros
    volume would yield NaNs.
    """
    size = 100
    vol = np.zeros((size, size, size), dtype=np.float32)
    zz, yy, xx = np.ogrid[:size, :size, :size]
    c = size // 2
    sphere = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 < (size // 3) ** 2
    vol[sphere] = 1.0
    return vol
