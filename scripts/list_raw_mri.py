"""List all downloaded raw MRI scans under the ADNI root and save a reference CSV.

ADNI on-disk layout: SUBJECT / PROTOCOL / ACQ_DATE / I<image_id> / <file>
Scans come in two formats: a single .nii file, or a DICOM series (many .dcm
slices in the I<id> folder). We emit ONE row per scan (per I<id> folder).

Usage (run from repo root):
    python scripts/list_raw_mri.py
    python scripts/list_raw_mri.py --root /mnt/d/lucas/Downloads/raw/ADNI \
        --output data/reference/DOWNLOAD_RAW_MRI.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd

IMAGE_DIR_RE = re.compile(r"^I\d+$")
# ADNI subject id, e.g. 002_S_0295 (site _ S _ id). Used to find the subject
# path segment by pattern instead of a fixed depth, so the result is correct
# regardless of where --root points relative to the subject folders.
SUBJECT_RE = re.compile(r"^\d{3}_S_\d{4}$")
IMAGE_EXTS = {".nii", ".dcm"}


def _to_sorted_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values(["SUBJECT", "IMAGE_DATA_ID"])
    return df.reset_index(drop=True)


def list_raw_mris(root: Path, output: Path, save_every: int = 1000) -> pd.DataFrame:
    """One row per scan, keyed on the I<id> folder.

    Checkpoints the CSV every `save_every` scans (and prints progress), so a
    long walk over the mounted drive isn't lost if it's interrupted.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_dir in root.rglob("I[0-9]*"):
        if not image_dir.is_dir() or not IMAGE_DIR_RE.match(image_dir.name):
            continue

        files = [f for f in image_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue

        # subject = the path segment matching the ADNI id pattern (NNN_S_NNNN).
        # Matching by pattern (not a fixed depth) keeps this correct whether
        # --root is the ADNI folder or one level above it. Fall back to the
        # first segment under root only if no segment matches.
        rel_parts = image_dir.relative_to(root).parts
        subject = next((p for p in rel_parts if SUBJECT_RE.match(p)), rel_parts[0])
        fmt = files[0].suffix.lower().lstrip(".")  # nii or dcm

        rows.append({
            "SUBJECT": subject,
            "IMAGE_DATA_ID": image_dir.name,      # e.g. I123685
            "IMAGE_NAME": files[0].name,           # representative file name
            "FORMAT": fmt,
            "N_FILES": len(files),                 # 1 for nii, many for dcm
            "PATH": str(image_dir),
        })

        if len(rows) % save_every == 0:
            _to_sorted_df(rows).to_csv(output, index=False)
            print(f"  ... {len(rows)} scans found (checkpoint saved -> {output})")

    df = _to_sorted_df(rows)
    df.to_csv(output, index=False)  # final save (covers the last partial batch)
    return df


def main():
    parser = argparse.ArgumentParser(description="List downloaded raw ADNI MRI scans.")
    parser.add_argument("--root", default="/mnt/d/lucas/Downloads/raw/ADNI",
                        help="Root ADNI download directory (subjects at first level).")
    parser.add_argument("--output", default="data/reference/DOWNLOAD_RAW_MRI.csv",
                        help="Output CSV path (run from repo root).")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    df = list_raw_mris(root, out)

    print(f"Scans found: {len(df)}")
    print(f"  unique subjects: {df['SUBJECT'].nunique()}")
    print(f"  by format:\n{df['FORMAT'].value_counts().to_string()}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
