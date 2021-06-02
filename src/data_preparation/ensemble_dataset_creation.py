# %%
import pandas as pd
import numpy as np


# %%
def create_final_ensemble_dataset(mri_dataset_path,cognitive_tests_dataset_path,output_path):

    # TODO: Implement coronal,axial and sagittal import

    df_mri = pd.read_csv(mri_dataset_path)
    df_mri.rename(columns={
        'IMAGE_DATA_ID':'IMAGEUID',
    },inplace=True)
    df_mri.drop(['SUBJECT_IMAGE_ID','UNIQUE_IMAGE_ID', 'SEX', 'AGE','TYPE'])
    df_cognitive = pd.read_csv(cognitive_tests_dataset_path)

    # TODO: import ensemble reference to concatenate with everything.
    # Or maybe just write MRI and tabular results on ensemble reference file....
    print("Aligning Ensemble dataset...")
    df_ensemble_dataset = pd.merge(df_cognitive,df_mri, on=['SUBJECT','IMAGEUID'],how='outer')
    print("Size:",df_ensemble_dataset.shape)

    print("Saving Ensemble dataset...")
    df_ensemble_dataset.to_csv(output_path + 'ENSEMBLE_DATASET.csv')
