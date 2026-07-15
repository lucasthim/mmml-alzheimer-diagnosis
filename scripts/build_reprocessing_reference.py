#!/usr/bin/env python3
"""Build a reference CSV of exactly which raw MRI scans still need (re)processing.

Why this exists
----------------
Two problems converged:

1. A batch of DICOM-series scans went through ``mri_preprocessing.py`` and came
   out corrupted, so every DICOM scan needs to be reprocessed regardless of
   whether a (bad) output already exists for it.
2. It's unclear which ``.nii`` scans have already been preprocessed successfully
   and which haven't, so those should only be (re)processed if no output exists yet.

This script answers both by combining three inputs:

- ``DOWNLOAD_RAW_MRI.csv`` (schema from ``scripts/list_raw_mri.py``): every raw
  scan on disk, one row per ``I<id>`` folder, tagged ``FORMAT`` = ``nii`` or ``dcm``.
- ``ADNIMERGE.csv``: the canonical list of scans that actually matter (its
  ``IMAGEUID`` column). Anything not in ADNIMERGE is out of scope.
- The preprocessed output tree (``*.nii.gz`` under ``--preprocessed-root``,
  searched recursively across every dated run folder): which ``IMAGE_DATA_ID``s
  already have output.

Selection rule, per ADNIMERGE-matched raw scan:

- ``FORMAT == dcm``  -> always keep (redo, output is presumed corrupted).
- ``FORMAT == nii``  -> keep only if its ``IMAGE_DATA_ID`` has no existing
  ``.nii.gz`` anywhere under ``--preprocessed-root``.

The output uses the same schema as ``DOWNLOAD_RAW_MRI.csv`` (plus a ``REASON``
column for provenance), so it can be fed straight into
``mri_preprocessing.py --reference-csv`` with no further filtering needed.

Usage (run from repo root)
---------------------------
    python scripts/build_reprocessing_reference.py

    python scripts/build_reprocessing_reference.py \
        --raw-reference data/reference/DOWNLOAD_RAW_MRI.csv \
        --adnimerge data/tabular/ADNIMERGE.csv \
        --preprocessed-root /mnt/d/lucas/Downloads/preprocessed \
        --output data/reference/REPROCESS_MRI_REFERENCE.csv

Then preprocess directly from the result:
    python src/data_preprocessing/mri_preprocessing.py \
        --reference-csv data/reference/REPROCESS_MRI_REFERENCE.csv \
        --output /mnt/d/lucas/Downloads/preprocessed/<today> -w 3
"""
import argparse
import re
from pathlib import Path

import pandas as pd

IMAGE_ID_FROM_OUTPUT_RE = re.compile(r"(I\d+)\.nii\.gz$")


def find_already_preprocessed_ids(preprocessed_root: Path) -> set:
    """IMAGE_DATA_IDs that already have a preprocessed .nii.gz somewhere under root.

    Searched recursively so every dated run folder (e.g. 20260707, 20260708, ...)
    counts, not just the latest one.
    """
    ids = set()
    for f in preprocessed_root.rglob("*.nii.gz"):
        m = IMAGE_ID_FROM_OUTPUT_RE.search(f.name)
        if m:
            ids.add(m.group(1))
    return ids


def build_reprocessing_reference(raw_reference_path, adnimerge_path, preprocessed_root, output_path) -> pd.DataFrame:
    df_raw = pd.read_csv(raw_reference_path)
    print(f"Raw reference: {len(df_raw)} scans ({raw_reference_path})")

    df_adni = pd.read_csv(adnimerge_path, low_memory=False)
    adni_ids = set('I' + df_adni['IMAGEUID'].dropna().astype(int).astype(str))
    print(f"ADNIMERGE: {len(adni_ids)} unique IMAGEUIDs ({adnimerge_path})")

    df_scoped = df_raw[df_raw['IMAGE_DATA_ID'].isin(adni_ids)].copy()
    print(f"Raw scans in scope (matched to ADNIMERGE): {len(df_scoped)}/{len(df_raw)}")

    preprocessed_root = Path(preprocessed_root)
    done_ids = find_already_preprocessed_ids(preprocessed_root)
    print(f"Already-preprocessed IMAGE_DATA_IDs found under {preprocessed_root}: {len(done_ids)}")

    is_dcm = df_scoped['FORMAT'].astype(str).str.lower() == 'dcm'
    already_done = df_scoped['IMAGE_DATA_ID'].isin(done_ids)

    keep = is_dcm | ~already_done
    df_out = df_scoped[keep].copy()
    df_out['ALREADY_PREPROCESSED'] = already_done[keep]
    df_out['REASON'] = 'nii_missing'
    df_out.loc[is_dcm[keep], 'REASON'] = 'dcm_reprocess'

    df_out = df_out.sort_values(['SUBJECT', 'IMAGE_DATA_ID']).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    n_dcm = int(is_dcm[keep].sum())
    n_nii = len(df_out) - n_dcm
    n_dcm_redo = int((is_dcm & already_done)[keep].sum())
    print('-------------------------------------------------------------')
    print(f"Scans to (re)process: {len(df_out)}")
    print(f"  DICOM (all kept, regardless of prior output): {n_dcm}  ({n_dcm_redo} had corrupted output to overwrite)")
    print(f"  NIfTI (kept only because no output yet):      {n_nii}")
    print(f"Saved -> {output_path}")
    print('-------------------------------------------------------------')
    return df_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw-reference', default='data/reference/DOWNLOAD_RAW_MRI.csv',
                         help='DOWNLOAD_RAW_MRI.csv-schema listing of every raw scan on disk (scripts/list_raw_mri.py).')
    parser.add_argument('--adnimerge', default='data/tabular/ADNIMERGE.csv',
                         help='ADNIMERGE.csv path — the canonical set of in-scope IMAGEUIDs.')
    parser.add_argument('--preprocessed-root', default='/mnt/d/lucas/Downloads/preprocessed',
                         help='Root folder to scan recursively for existing *.nii.gz outputs (all dated run subfolders).')
    parser.add_argument('--output', default='data/reference/REPROCESS_MRI_REFERENCE.csv',
                         help='Output reference CSV path.')
    args = parser.parse_args()

    build_reprocessing_reference(
        raw_reference_path=args.raw_reference,
        adnimerge_path=args.adnimerge,
        preprocessed_root=args.preprocessed_root,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
