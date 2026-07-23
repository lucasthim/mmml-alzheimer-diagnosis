"""
Generate the 6 training slices (3D -> 2D) for every preprocessed MRI, following the
2026 Training Runbook (docs/experiments/2026-training-runbook.md, Steps 5b + 8).

Why a runner script: mri_batch_preparation.py has no CLI (its __main__ is commented out).
This script (a) builds the enriched PREPROCESSED_MRI_REFERENCE.csv the slicer needs, then
(b) slices all 6 indices in one call, then (c) backfills the DATASET column.

Runbook-critical details baked in here:
  - Labels (SUBJECT/GROUP/MACRO_GROUP/SEX/AGE) are backfilled from COGNITIVE_DATA_PREPROCESSED.csv
    (0 missing IMAGE_DATA_IDs), NOT REFERENCE_TABLE_FOR_MRI.csv (75 missing, stale).
  - MACRO_GROUP is the STRING label 'CN'/'AD'/'MCI' (mapped from DIAGNOSIS 0/1/2). Numeric
    MACRO_GROUP would make return_sets()'s MCI remap silently no-op -> an MCIxCN run would
    keep AD rows instead of MCI rows.
  - One combined call: {'coronal':[43,70], 'axial':[23,8], 'sagittal':[26,50]} loads each
    volume 3x (once per orientation) instead of 6x. Unique dict keys -> no duplicate-key bug.
  - After slicing, DATASET==NaN rows (images outside the ensemble's cognitively-complete
    cohort) are set to 'train', giving the CNN its full training cohort. validation/test are
    untouched, so all models share the same eval set.

Run in the background with output logged to a file:
    cd /home/lucasthim/projects/phd/mmml-alzheimer-diagnosis
    nohup uv run python -u src/data_preparation/run_slice_preparation.py \
        > data/mri/experiments/slice_prep_$(date +%Y%m%d_%H%M).log 2>&1 &
"""
import os
import sys

import pandas as pd

REPO_ROOT = "/home/lucasthim/projects/phd/mmml-alzheimer-diagnosis"

# mri_batch_preparation's module-level imports (`from mri_augmentation import *`,
# `sys.path.append("./../utils")`) are CWD-relative -- CWD must be src/data_preparation.
os.chdir(os.path.join(REPO_ROOT, "src/data_preparation"))
sys.path.insert(0, os.getcwd())

from mri_batch_preparation import execute_mri_batch_preparation
    
# ---- inputs -----------------------------------------------------------------
NEW_REFERENCE = "/mnt/d/lucas/Downloads/preprocessed/20260722/REFERENCE.csv"   # this atlas-fixed run
COGNITIVE     = os.path.join(REPO_ROOT, "data/tabular/COGNITIVE_DATA_PREPROCESSED.csv")
ENSEMBLE_REF  = os.path.join(REPO_ROOT, "data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv")  # DATASET split + CONFLICT

# enriched reference the slicer consumes; MUST be named PREPROCESSED_MRI_REFERENCE.csv so the
# module's output PROCESSED_MRI_REFERENCE_<ts>.csv lands in data/reference/ (naming logic at :97).
ENRICHED_REF  = os.path.join(REPO_ROOT, "data/reference/PREPROCESSED_MRI_REFERENCE.csv")


SLICE_OUTPUT  = os.path.join(REPO_ROOT, "data/mri/processed/storage/")

# ---- the 6 slices (one list per orientation key -> 3 volume loads/image, no dup-key bug) ----
ORIENTATIONS = {
    "coronal":  [43, 70],   # AD 43, MCI 70
    "axial":    [23, 8],    # AD 23, MCI 8
    "sagittal": [26, 50],   # AD 26, MCI 50
}

LABEL_MAP = {0: "CN", 1: "AD", 2: "MCI"}   # DIAGNOSIS -> string MACRO_GROUP (runbook: must be string)


def build_enriched_reference():
    """Runbook Step 5b: backfill labels from cognitive data, drop rows whose file is gone."""
    df = pd.read_csv(NEW_REFERENCE)
    print(f"Preprocessing REFERENCE.csv: {len(df)} images", flush=True)

    df_cog = pd.read_csv(COGNITIVE, low_memory=False)
    df_cog = df_cog.dropna(subset=["IMAGEUID"]).copy()
    df_cog["IMAGE_DATA_ID"] = "I" + df_cog["IMAGEUID"].astype(int).astype(str)
    df_cog["MACRO_GROUP"] = df_cog["DIAGNOSIS"].map(LABEL_MAP)   # string labels
    df_cog["GROUP"] = df_cog["MACRO_GROUP"]                       # unused downstream, schema only
    df_cog["SEX"] = df_cog["MALE"]                                 # unused downstream, schema only
    cog_cols = ["IMAGE_DATA_ID", "SUBJECT", "GROUP", "MACRO_GROUP", "SEX", "AGE"]
    df_cog = df_cog[cog_cols].drop_duplicates(subset="IMAGE_DATA_ID")

    df = df.merge(df_cog, on="IMAGE_DATA_ID", how="left")
    n_nolabel = int(df["MACRO_GROUP"].isna().sum())

    before = len(df)
    df = df[df["IMAGE_PATH"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Enrichment: {len(df)}/{before} files exist on disk "
          f"({before - len(df)} missing dropped); {n_nolabel} rows with no cognitive-data match.",
          flush=True)

    os.makedirs(os.path.dirname(ENRICHED_REF), exist_ok=True)
    df.to_csv(ENRICHED_REF, index=False)
    print(f"Wrote enriched reference ({len(df)} rows) -> {ENRICHED_REF}", flush=True)
    return ENRICHED_REF


def backfill_dataset_and_write_master(out_ref):
    """Runbook Step 8 extension: NaN DATASET (outside cognitively-complete cohort) -> 'train'.
    validation/test untouched. Writes the ALL_ORIENTATIONS master training reference."""
    df = pd.read_csv(out_ref)
    fillable = df["DATASET"].isna() & df["MACRO_GROUP"].notna()
    df.loc[fillable, "DATASET"] = "train"
    print(f"DATASET backfill: set {int(fillable.sum())} rows to 'train'. "
          f"Counts: {df['DATASET'].value_counts(dropna=False).to_dict()}", flush=True)

    master = out_ref.replace("PROCESSED_MRI_REFERENCE_", "PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_")
    df.to_csv(master, index=False)
    print(f"Wrote training master reference -> {master}", flush=True)
    return master


def main():
    print("=" * 90, flush=True)
    print("SLICE PREPARATION (2026 runbook Step 8):", ORIENTATIONS, flush=True)
    print("=" * 90, flush=True)

    mri_reference_path = build_enriched_reference()

    out_ref = execute_mri_batch_preparation(
        mri_reference_path=mri_reference_path,
        ensemble_reference_path=ENSEMBLE_REF,
        output_path=SLICE_OUTPUT,
        orientations=ORIENTATIONS,
    )

    master = backfill_dataset_and_write_master(out_ref)

    print("\n" + "=" * 90, flush=True)
    print(f"DONE. Slices -> {SLICE_OUTPUT}", flush=True)
    print(f"Point train_all_cnns.py's MRI_REFERENCE at -> {master}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
