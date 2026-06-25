"""Smoke tests for deepbrain skull stripping under TensorFlow 2.x.

These verify the ``tf.compat.v1`` patch on the deepbrain fork: that the frozen U-Net
graph loads and runs on TF 2.x and that the project's ``deep_brain_skull_stripping``
wrapper produces sane output. They use a synthetic dummy volume (no real MRI data) and
are device-agnostic (pass on CPU and GPU).
"""
import numpy as np
import pytest

# deepbrain_skull_strip imports deepbrain (TensorFlow) before ants, in that order, to avoid
# the macOS OpenMP deadlock — import it before touching ants here.
from deepbrain_skull_strip import deep_brain_skull_stripping
from deepbrain import Extractor
import ants
from ants.core.ants_image import ANTsImage

TOL = 1e-3  # skimage resize anti-aliasing can marginally overshoot [0, 1]


def test_extractor_runs(dummy_volume):
    """Core regression test for the compat.v1 patch.

    If the patch were wrong, ``Extractor()`` raises at construction; reaching a finite
    probability map of the right shape proves the frozen graph executes on TF 2.x.
    """
    prob = Extractor().run(dummy_volume)

    assert prob.shape == dummy_volume.shape
    assert np.issubdtype(prob.dtype, np.floating)
    assert np.isfinite(prob).all()
    assert prob.min() >= -TOL and prob.max() <= 1.0 + TOL


def test_skull_strip_array(dummy_volume):
    stripped = deep_brain_skull_stripping(dummy_volume, probability=0.5, output_as_array=True)

    assert isinstance(stripped, np.ndarray)
    assert stripped.shape == dummy_volume.shape
    assert np.isfinite(stripped).all()


def test_skull_strip_get_mask(dummy_volume):
    mask = deep_brain_skull_stripping(dummy_volume, probability=0.5, get_mask=True)

    assert mask.shape == dummy_volume.shape
    assert set(np.unique(mask.astype(int))).issubset({0, 1})


def test_skull_strip_ants(dummy_volume):
    """Exercises the ANTsImage input branch (direction handling)."""
    img = ants.from_numpy(dummy_volume)
    stripped = deep_brain_skull_stripping(img, probability=0.5, output_as_array=False)

    assert isinstance(stripped, ANTsImage)
    assert tuple(stripped.shape) == dummy_volume.shape
