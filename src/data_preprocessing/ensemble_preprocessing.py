import argparse

import pandas as pd
import numpy as np


def execute_ensemble_preprocessing(preprocessed_cognitive_data_path,
                            ensemble_data_output_path,
                            classes=[0,1],
                            downloaded_mri_reference_path=None):
    '''

    Build the ensemble reference table from the preprocessed cognitive data.

    2026 rewrite
    ------------
    The 2026 ADNIMERGE already carries the image code (IMAGEUID) and the diagnosis
    (DX -> DIAGNOSIS) in a single table, so COGNITIVE_DATA_PREPROCESSED.csv is a
    sufficient source on its own. The original two-source design — merging a
    SEPARATELY downloaded MRI metadata table and reconciling its GROUP label
    against the cognitive DX — no longer applies: there is only one diagnosis
    source now. That merge + conflict reconciliation has been removed.

    What this produces is unchanged in SCHEMA, so the downstream contract holds
    (ensemble_preparation.py + the mri_*_preparation.py modules read these):
      - SUBJECT              (str)   subject id
      - IMAGEUID             (int)   ADNI image id, no 'I' prefix
      - DIAGNOSIS            (int)   0/1/2 label (CN/AD/MCI)
      - MACRO_GROUP          (int)   0/1/2 — now equal to DIAGNOSIS (single source)
      - CONFLICT_DIAGNOSIS   (bool)  always False now (kept so downstream filters run)
      - all cognitive feature columns, passed through

    Steps:
    1. Read cognitive data; drop NaN rows and no-MRI rows (IMAGEUID == 999999).
    2. Set MACRO_GROUP = DIAGNOSIS and CONFLICT_DIAGNOSIS = False (contract columns).
    3. Drop the hardcoded missing-axial-MRI blacklist.
    4. Filter to the requested class pair.
    5. (Optional) tag each row with HAS_PREPROCESSED_MRI from the downloaded-MRI
       reference (DOWNLOAD_RAW_MRI.csv). This is an EXTRA, non-breaking column;
       rows are NOT filtered on it here.
    6. Save the ensemble reference file.

    Parameters
    ----------
    preprocessed_cognitive_data_path: path to COGNITIVE_DATA_PREPROCESSED.csv.
    ensemble_data_output_path: path to write PREPROCESSED_ENSEMBLE_REFERENCE.csv.
    classes: class pair to keep (default [0,1] = CN vs AD). MCI is 2.
    downloaded_mri_reference_path: optional DOWNLOAD_RAW_MRI.csv path. When given,
        adds a HAS_PREPROCESSED_MRI flag; does not filter rows.
    '''

    print("Reading preprocessed cognitive data...")
    df_ensemble = pd.read_csv(preprocessed_cognitive_data_path).dropna().query("IMAGEUID != 999999").copy()
    print(f"Cognitive rows with a real MRI id: {df_ensemble.shape[0]}")

    # 2026: diagnosis has a single source (ADNIMERGE DX -> DIAGNOSIS). The MRI-side
    # label and the two-source conflict check are gone, so MACRO_GROUP mirrors
    # DIAGNOSIS and every row is conflict-free by construction. Both columns are
    # kept because ensemble_preparation.py and the mri_*_preparation.py modules
    # still read them.
    df_ensemble['MACRO_GROUP'] = df_ensemble['DIAGNOSIS']
    df_ensemble['CONFLICT_DIAGNOSIS'] = False

    # TODO: I think this function below is only relevant to the old data sources.
    # df_ensemble = remove_missing_mris_in_validation(df_ensemble)
    
    df_ensemble = df_ensemble.query("MACRO_GROUP in @classes")
    print(f"Rows after filtering to classes {classes}: {df_ensemble.shape[0]}")

    if downloaded_mri_reference_path is not None:
        df_ensemble = tag_preprocessed_mri_availability(df_ensemble, downloaded_mri_reference_path)

    print("Saving ensemble reference file...")
    df_ensemble.to_csv(ensemble_data_output_path, index=False)
    return df_ensemble


def tag_preprocessed_mri_availability(df_ensemble, downloaded_mri_reference_path):
    '''Add a HAS_PREPROCESSED_MRI bool: True when the row's IMAGEUID has a scan
    listed in the downloaded-MRI reference (DOWNLOAD_RAW_MRI.csv, IMAGE_DATA_ID =
    'I' + IMAGEUID). Does not drop any rows.'''
    df_dl = pd.read_csv(downloaded_mri_reference_path)
    downloaded_ids = set(df_dl['IMAGE_DATA_ID'].astype(str).str.replace('I', '', regex=False).astype(np.int64))
    df_ensemble['HAS_PREPROCESSED_MRI'] = df_ensemble['IMAGEUID'].astype(np.int64).isin(downloaded_ids)
    n = int(df_ensemble['HAS_PREPROCESSED_MRI'].sum())
    print(f"Tagged HAS_PREPROCESSED_MRI: {n}/{df_ensemble.shape[0]} rows have a downloaded scan.")
    return df_ensemble


def remove_missing_mris_in_validation(df_ensemble):
    missing_axial_mris_validation = [293688, 274525, 280596]
    return df_ensemble.query("IMAGEUID not in @missing_axial_mris_validation")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Build the ensemble reference table from preprocessed cognitive data (2026 single-source ADNIMERGE).')

    arg_parser.add_argument('-c', '--cognitive',
                        metavar='preprocessed_cognitive_data_path',
                        type=str,
                        required=False,
                        default='data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                        help='Path to COGNITIVE_DATA_PREPROCESSED.csv. Default: data/tabular/COGNITIVE_DATA_PREPROCESSED.csv')

    arg_parser.add_argument('-o', '--output',
                        metavar='ensemble_data_output_path',
                        type=str,
                        required=False,
                        default='data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv',
                        help='Output path for the ensemble reference CSV. Default: data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv')

    arg_parser.add_argument('--classes',
                        metavar='classes',
                        type=int,
                        nargs='+',
                        required=False,
                        default=[0, 1],
                        help='Class pair to keep (CN=0, AD=1, MCI=2). Default: 0 1 (CN vs AD).')

    arg_parser.add_argument('-d', '--downloaded-mri-reference',
                        metavar='downloaded_mri_reference_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Optional DOWNLOAD_RAW_MRI.csv path. When given, adds a HAS_PREPROCESSED_MRI '
                             'flag (does not filter rows).')

    args = arg_parser.parse_args()

    execute_ensemble_preprocessing(
        preprocessed_cognitive_data_path=args.cognitive,
        ensemble_data_output_path=args.output,
        classes=args.classes,
        downloaded_mri_reference_path=args.downloaded_mri_reference,
    )
