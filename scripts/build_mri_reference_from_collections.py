#!/usr/bin/env python3
"""Build ``REFERENCE_TABLE_FOR_MRI.csv`` — the per-image MRI metadata reference —
from the LONI image-collection CSV exports.

Why this exists
---------------
The MRI pipeline needs a "previous reference" table (historically
``MRI_MPRAGE.csv`` / ``RAW_MRI_REFERENCE.csv``) carrying
``SUBJECT, GROUP, MACRO_GROUP, SEX, AGE`` per image, which
``src/utils/utils.py::create_reference_table`` merges into the preprocessed
``REFERENCE.csv`` on ``IMAGE_DATA_ID`` (via ``load_reference_table``).
``mri_preprocessing.py`` with no ``-r`` only writes the 4 filename-derived columns,
so that metadata is otherwise missing.

That metadata lives in the **LONI image-collection CSV exports** (the per-collection
"CSV" download), columns ``Image Data ID, Subject, Group, Sex, Age, Visit,
Modality, Description, Type, Acq Date, Format, Downloaded``.

This script just normalizes and concatenates those exports into the reference the
pipeline expects — **no dependency on the downloaded ``.nii`` files**. The image
collections were themselves selected from ``ADNIMERGE.csv`` (built by
``rebuild_adnimerge_from_adnimerge2.py``), so the ``Image Data ID``s here are the
canonical keys: images you download *from these collections* carry the same IDs,
so this reference key-matches the preprocessed ``REFERENCE.csv`` with no bridging.

It is effectively a clean reimplementation of
``mri_metadata_preprocessing.execute_mri_metadata_preprocessing_prior_to_image_preprocessing``
(concat the exports, drop the always-constant columns, dedup on ``IMAGE_DATA_ID``)
plus ``load_reference_table``'s ``MACRO_GROUP`` / ``SUBJECT_IMAGE_ID`` derivation.

Usage
-----
    python scripts/build_mri_reference_from_collections.py \
        --metadata data/tabular --prefix MRI_2026 \
        --out data/reference/REFERENCE_TABLE_FOR_MRI.csv --selfcheck
"""
import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# GROUP -> MACRO_GROUP, mirroring src/utils/utils.py::load_reference_table
MACRO_MAP = {"SMC": "CN", "EMCI": "MCI", "LMCI": "MCI"}

# Always-constant / non-informative columns the production metadata step drops
# (mri_metadata_preprocessing.py drop_cols).
DROP_COLS = ["MODALITY", "TYPE", "FORMAT", "DOWNLOADED", "UNIQUE_IMAGE_ID"]

# Final column order (uppercase, what load_reference_table / mri_batch_preparation read)
OUT_COLS = ["SUBJECT_IMAGE_ID", "IMAGE_DATA_ID", "SUBJECT", "GROUP", "MACRO_GROUP",
            "SEX", "AGE", "VISIT", "ACQ_DATE", "DESCRIPTION"]


def load_collection_exports(metadata_dir, prefix, verbose=1):
    """Concatenate the LONI image-collection CSV exports (``<prefix>*.csv``)."""
    files = sorted(glob.glob(os.path.join(metadata_dir, prefix + "*.csv")))
    if not files:
        sys.exit(f"No metadata CSVs matching '{prefix}*.csv' in {metadata_dir}")
    if verbose > 0:
        print(f"  exports: {[os.path.basename(f) for f in files]}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # same column-normalization idiom as load_reference_table: spaces->_, upper
    df.columns = [str(c).replace(" ", "_").upper() for c in df.columns]
    return df


def create_reference_table_from_metadata(metadata_dir, out_path, prefix="MRI_2026",
                                         drop_masks=False, verbose=1):
    """Build the per-image MRI reference from the collection CSV exports.

    Params
    ------
    metadata_dir : directory holding the ``<prefix>*.csv`` collection exports
    out_path     : output CSV path (``REFERENCE_TABLE_FOR_MRI.csv``)
    prefix       : metadata-CSV filename prefix (default ``MRI_2026``)
    drop_masks   : drop rows whose DESCRIPTION contains 'Mask' (preprocessing
                   excludes masks anyway; off by default for fidelity)
    """
    df = load_collection_exports(metadata_dir, prefix, verbose)

    required = ["IMAGE_DATA_ID", "SUBJECT", "GROUP"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Exports missing required columns {missing}; got {list(df.columns)}")

    # drop always-constant columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if drop_masks and "DESCRIPTION" in df.columns:
        before = len(df)
        df = df[~df["DESCRIPTION"].astype(str).str.contains("Mask", case=False, na=False)]
        if verbose > 0:
            print(f"  dropped {before - len(df)} mask rows")

    # normalize Acq Date to ISO where parseable
    if "ACQ_DATE" in df.columns:
        dt = pd.to_datetime(df["ACQ_DATE"], format="mixed", errors="coerce")
        df["ACQ_DATE"] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), df["ACQ_DATE"])

    # one row per image
    df = df.drop_duplicates("IMAGE_DATA_ID", keep="first").reset_index(drop=True)

    # derived columns (match load_reference_table)
    df["MACRO_GROUP"] = df["GROUP"].replace(MACRO_MAP)
    df["SUBJECT_IMAGE_ID"] = df["SUBJECT"].astype(str) + "#" + df["IMAGE_DATA_ID"].astype(str)

    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    out = df[OUT_COLS].sort_values(["SUBJECT", "ACQ_DATE", "IMAGE_DATA_ID"])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out.to_csv(out_path, index=False)
    if verbose > 0:
        print(f"\nWrote {len(out)} rows x {len(out.columns)} cols -> {out_path}")
    return out


def selfcheck(df):
    """Validate the load_reference_table column contract and report coverage."""
    print("\n=== SELF-CHECK ===")
    print(f"  rows: {len(df)}   distinct images: {df['IMAGE_DATA_ID'].nunique()}"
          f"   subjects: {df['SUBJECT'].nunique()}")
    print(f"  duplicate IMAGE_DATA_ID rows: {int(df['IMAGE_DATA_ID'].duplicated().sum())}")
    for col in ("IMAGE_DATA_ID", "SUBJECT", "GROUP"):  # load_reference_table needs these
        ok = col in df.columns and df[col].notna().any()
        print(f"  load_reference_table needs '{col}': {'OK' if ok else 'MISSING'}")
    print(f"  GROUP distribution:       {df['GROUP'].value_counts(dropna=False).to_dict()}")
    print(f"  MACRO_GROUP distribution: {df['MACRO_GROUP'].value_counts(dropna=False).to_dict()}")
    print(f"  SEX distribution:         {df['SEX'].value_counts(dropna=False).to_dict()}")
    nmask = int(df["DESCRIPTION"].astype(str).str.contains("Mask", case=False, na=False).sum())
    print(f"  rows describing a Mask (excluded by preprocessing anyway): {nmask}")
    parsed = pd.to_datetime(df["ACQ_DATE"], errors="coerce")
    if parsed.notna().any():
        print(f"  ACQ_DATE range: {parsed.min().date()} -> {parsed.max().date()}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/tabular",
                    help="directory holding the <prefix>*.csv collection exports")
    ap.add_argument("--prefix", default="MRI_2026",
                    help="filename prefix of the collection-export CSVs")
    ap.add_argument("--out", default="data/reference/REFERENCE_TABLE_FOR_MRI.csv",
                    help="output reference CSV path")
    ap.add_argument("--drop-masks", action="store_true",
                    help="drop rows whose DESCRIPTION contains 'Mask'")
    ap.add_argument("--selfcheck", action="store_true",
                    help="print coverage + column-contract validation")
    ap.add_argument("-v", "--verbose", type=int, default=1, help="0=quiet, 1=normal")
    args = ap.parse_args()

    print(f"Building {os.path.basename(args.out)} from {args.prefix}*.csv "
          f"in {args.metadata} ...")
    out = create_reference_table_from_metadata(
        metadata_dir=args.metadata, out_path=args.out, prefix=args.prefix,
        drop_masks=args.drop_masks, verbose=args.verbose)
    if args.selfcheck:
        selfcheck(out)


if __name__ == "__main__":
    main()
