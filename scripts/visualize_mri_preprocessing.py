#!/usr/bin/env python
"""Visualize center slices of an MRI, comparing a raw scan against its preprocessed result.

Mirrors the ``show_brain_center_slice`` check from
``notebooks/early_mri_exploration/03_MRI_preprocessing.ipynb``: for each volume it shows the
three center slices (sagittal, coronal, axial). Given both a raw and a preprocessed image it
stacks them as two rows so you can eyeball whether registration + skull stripping + cropping
worked.

Loads NIfTI (``.nii`` / ``.nii.gz``) via nibabel and 2D/3D ``.npz`` (key ``arr_0``) via numpy.
It only reads and plots, so it needs neither ANTsPy nor TensorFlow.

Examples
--------
Compare a raw scan with its preprocessed output and save a PNG:

    python scripts/visualize_mri_preprocessing.py \
        --raw  data/ADNI/.../ADNI_..._I258686.nii \
        --preprocessed data/mri/preprocessed/20260625/ADNI_..._I258686.nii.gz \
        --output reports/figures/I258686_check.png

Just look at a single volume on screen:

    python scripts/visualize_mri_preprocessing.py --raw data/ADNI/.../scan.nii --show
"""
import argparse
import os

import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt

VIEW_NAMES = ("Sagittal", "Coronal", "Axial")


def load_volume(path: str) -> np.ndarray:
    """Load an MRI volume as a 3D numpy array (no ANTs/TF dependency)."""
    if path.endswith(".npz"):
        return np.load(path)["arr_0"]
    if path.endswith((".nii", ".nii.gz")):
        return np.asarray(nib.load(path).get_fdata())
    raise ValueError(f"Unsupported file type: {path} (expected .nii, .nii.gz or .npz)")


def center_slices(volume: np.ndarray):
    """Return the three center slices (sagittal, coronal, axial) of a 3D volume."""
    c0, c1, c2 = (s // 2 for s in volume.shape)
    return [volume[c0, :, :], volume[:, c1, :], volume[:, :, c2]]


def _plot_row(axes, volume: np.ndarray, row_label: str):
    """Draw the three center slices of ``volume`` onto a row of axes (matches the notebook)."""
    for ax, sl, view in zip(axes, center_slices(volume), VIEW_NAMES):
        ax.imshow(sl.T, cmap="gray", origin="lower")
        ax.set_title(view, fontsize=12)
    axes[0].set_ylabel(f"{row_label}\n{volume.shape}", fontsize=13, rotation=90,
                       labelpad=12, va="center")


def visualize(raw_path: str = None, preprocessed_path: str = None,
              output: str = None, show: bool = False):
    """Plot center slices for a raw and/or preprocessed MRI as stacked rows."""
    rows = []  # (label, path)
    if raw_path:
        rows.append(("Raw", raw_path))
    if preprocessed_path:
        rows.append(("Preprocessed", preprocessed_path))
    if not rows:
        raise ValueError("Provide at least one of --raw / --preprocessed.")

    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 4 * len(rows)), squeeze=False)
    for r, (label, path) in enumerate(rows):
        volume = load_volume(path)
        if volume.ndim != 3:
            raise ValueError(f"{path} is not 3D (shape {volume.shape}); cannot take center slices.")
        _plot_row(axes[r], volume, label)
        print(f"{label:13s} {os.path.basename(path)}  shape={volume.shape}")

    fig.suptitle("MRI center slices", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"Saved figure -> {output}")
    if show:
        plt.show()
    if not output and not show:
        print("Nothing displayed or saved: pass --output PATH and/or --show.")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize raw vs preprocessed MRI center slices (sagittal/coronal/axial).")
    parser.add_argument("-r", "--raw", type=str, default=None,
                        help="Path to the raw MRI (.nii/.nii.gz/.npz).")
    parser.add_argument("-p", "--preprocessed", type=str, default=None,
                        help="Path to the preprocessed MRI (.nii.gz/.npz).")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Where to save the figure as PNG. Created if missing.")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive window (needs a GUI backend).")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # headless-safe when only saving

    visualize(raw_path=args.raw, preprocessed_path=args.preprocessed,
              output=args.output, show=args.show)
