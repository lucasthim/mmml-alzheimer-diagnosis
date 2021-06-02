# %%
import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import pandas as pd

sys.path.append("./../utils/")
# from src.utils.utils import *
from utils import load_reference_table,list_available_images

# %%

def execute_mri_metadata_preprocessing(input = None,output = None):
    
    '''
    MRI Metadata Preprocessing pipeline.
    
    Main steps:
    
    - Merge MRI reference (metadata) files from different folders.
    
    - Create a unique subject_image ID.
    
    - Drop potential duplicates.

    - Save the preprocessed reference file in the main MRI folder.
    

    Parameters
    ----------
    
    input_path: path or list where metadata files are located.
    
    output_path: path to save preprocessed file.
    
    '''   

    if type(input) == str:
        print("Reading reference files from :",input)
        files = Path(input).rglob('*.csv')
        files = [x.as_posix() for x in files]
    if type(input) == list:
        files = input

    print(f"Found {len(files)} files to concatenate...")
    concat_df = []
    for file in files:
        print("Reading ",file)
        concat_df.append(load_reference_table(file))
    print('')
    df_reference = pd.concat(concat_df)
    df_reference.reset_index(drop=True,inplace=True)
    
    for col in ['FORMAT','TYPE','UNIQUE_IMAGE_ID','IMAGE_PATH','MODALITY','DOWNLOADED']:
        if col in df_reference.columns:
            df_reference.drop(col,axis=1,inplace=True)
    
    print("Removing potential duplicates...")
    duplicated = df_reference[['SUBJECT','IMAGE_DATA_ID']].duplicated(keep='first')
    df_reference = df_reference[~duplicated]


    print("Scanning raw mri files...")


    print("Saving concatenated reference file at:",output)
    df_reference.to_csv(output,index=False)
    return df_reference
    
# %%
if __name__ == '__main__':
    input = ['/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/MPRAGE_REFERENCE.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_CN_AD.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_01.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_02.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_03.csv'
            ]
    output = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/RAW_MRI_REFERENCE.csv'
    df = execute_mri_metadata_preprocessing(input,output)


# %%
