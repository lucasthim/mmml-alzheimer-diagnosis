# %%
import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import pandas as pd

# %load_ext autoreload
# %autoreload 2

sys.path.append("./../utils/")
# from src.utils.utils import *
from utils import load_reference_table

# %%

def execute_mri_metadata_preprocessing_prior_to_image_preprocessing():
    input = ['/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/MPRAGE_REFERENCE.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_CN_AD.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_01.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_02.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/REFERENCE_MRI_ENSEMBLE_03.csv'
            ]
    output = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/RAW_MRI_REFERENCE.csv'
    
    df = execute_mri_metadata_preprocessing(input,output,drop_cols=['FORMAT','TYPE','UNIQUE_IMAGE_ID','MODALITY','DOWNLOADED'])
    
    print("Preprocessed MRI metadata reference:",df.shape)
    print("Saving concatenated reference file at:",output)
    df.to_csv(output,index=False)
    return df

def execute_mri_metadata_preprocessing_after_image_preprocessing():
    input = ['/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/preprocessed/20210523/REFERENCE.csv',
            '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/preprocessed/20210602/REFERENCE.csv']
    output = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    
    df = execute_mri_metadata_preprocessing(input,output,drop_cols=['FORMAT','TYPE','UNIQUE_IMAGE_ID','MODALITY','DOWNLOADED','SUBJECT_ID'])
    
    print("Preprocessed MRI metadata reference:",df.shape)
    print("Saving concatenated reference file at:",output)
    df.to_csv(output,index=False)
    return df

def execute_mri_metadata_preprocessing(input,output,drop_cols=['FORMAT','TYPE','UNIQUE_IMAGE_ID','MODALITY','DOWNLOADED']):
    
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
    
    for col in drop_cols:
        if col in df_reference.columns:
            df_reference.drop(col,axis=1,inplace=True)
    if 'IMAGE_PATH' in df_reference.columns:
        df_reference = df_reference.query("IMAGE_PATH == IMAGE_PATH")
    
    # print("Removing duplicates...")
    # if 'IMAGE_PATH' in df_reference.columns:
    #     duplicated = df_reference[['IMAGE_DATA_ID']].duplicated(keep=False)
    #     print(f"Found a total of {duplicated.sum() / 2} duplicates")
    #     duplicated = df_reference[duplicated]['IMAGE_PATH'].isnull()
    #     df_reference = df_reference[~duplicated]
    #     duplicated = df_reference[['IMAGE_DATA_ID']].duplicated(keep='first')
    #     df_reference = df_reference[~duplicated]
    
    # else:
    duplicated = df_reference[['IMAGE_DATA_ID']].duplicated(keep='first')
    df_reference = df_reference[~duplicated]
    print(f"Found a total of {duplicated.sum()} duplicates")

    return df_reference

arg_parser = argparse.ArgumentParser(description='Select MRIs to download.')

arg_parser.add_argument('-t','--type',
                    metavar='mri_type',
                    type=str,
                    required=False,
                    default='raw',
                    choices=['raw','preprocessed']
                    help='Indicate if it is processing the metadata before (raw) or after image preprocessing.')

# %%
if __name__ == '__main__':

    if args.mri_type == 'raw':
        df1 = execute_mri_metadata_preprocessing_prior_to_image_preprocessing()

    elif args.mri_type == 'preprocessed':
        df2 = execute_mri_metadata_preprocessing_after_image_preprocessing()
    
# %%
# duplicated = df2[['IMAGE_DATA_ID']].duplicated(keep=False)
# duplicated = df2[duplicated]['IMAGE_PATH'].isnull()
# %%
