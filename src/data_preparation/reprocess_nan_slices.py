"""
Re-slice the 1,940 images that had >=1 NaN (missing) 2D slice in
PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20260710_1413.csv.

Regenerates the FULL 6-slice set (axial 8/23, coronal 43/70, sagittal 26/50) for
each of those images, writing .npz slices into the per-image storage folders and a
fresh PROCESSED_MRI_REFERENCE_<timestamp>.csv reference beside the input.

Inputs (already built):
  - data/reference/reprocess_nan_20260716/PREPROCESSED_MRI_REFERENCE.csv
      one row per 3D image to re-slice; IMAGE_PATH points at the 3D .nii.gz volume.
  - data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv
      supplies CONFLICT_DIAGNOSIS (filter) + DATASET (train/val/test) per image.

Output:
  - .npz slices under data/mri/processed/storage/<IMAGE_DATA_ID>/<orient>_<NN>.npz
    (empty slices are skipped by the slicer's validate_slice, exactly as before)
  - data/reference/reprocess_nan_20260716/PROCESSED_MRI_REFERENCE_<YYYYMMDD_HHMM>.csv

Run from the repo root:
    cd /home/lucasthim/projects/phd/mmml-alzheimer-diagnosis
    .venv/bin/python src/data_preparation/reprocess_nan_slices.py

Or in the background with a log:
    nohup .venv/bin/python -u src/data_preparation/reprocess_nan_slices.py \
        > data/reference/reprocess_nan_20260716/run_$(date +%Y%m%d_%H%M).log 2>&1 &
"""
import os
import sys

REPO_ROOT = "/home/lucasthim/projects/phd/mmml-alzheimer-diagnosis"

# mri_batch_preparation.py uses CWD-relative imports (`sys.path.append("./../utils")`),
# so CWD must be src/data_preparation/ for base_mri / utils / mri_augmentation to resolve.
os.chdir(os.path.join(REPO_ROOT, "src/data_preparation"))
sys.path.insert(0, os.getcwd())

from mri_batch_preparation import execute_mri_batch_preparation

MRI_REFERENCE = os.path.join(REPO_ROOT, "data/reference/reprocess_nan_20260716/PREPROCESSED_MRI_REFERENCE.csv")
ENSEMBLE_REFERENCE = os.path.join(REPO_ROOT, "data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data/mri/processed/storage/")

# Full 6-slice set. Explicit single-key-per-orientation lists avoid the dict-key
# collision bug in the module's default `orientations` argument.
ORIENTATIONS = {
    "coronal":  [43, 70],
    "axial":    [8, 23],
    "sagittal": [26, 50],
}


def main():
    out = execute_mri_batch_preparation(
        mri_reference_path=MRI_REFERENCE,
        ensemble_reference_path=ENSEMBLE_REFERENCE,
        output_path=OUTPUT_PATH,
        orientations=ORIENTATIONS,
    )
    print("\nDONE. New reference written to:", out)


if __name__ == "__main__":
    main()
