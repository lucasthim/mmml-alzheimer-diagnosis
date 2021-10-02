# %%
import pandas as pd
import numpy as np


# %%
def execute_ensemble_preprocessing(preprocessed_cognitive_data_path,preprocessed_mri_raw_data_path,ensemble_data_output_path,classes=[1,0],how='inner'):
    ''' 

    Execute preprocessing for the reference data that will be used in the ensemble.

    Main steps are:
    1. Remove duplicates from MRI reference (sometimes the same image is downloaded twice)
    2. Merge files from MRI reference and Cognitive Tests reference.
    3. Remove conflicting diagnosis between MRI and Congnitive Tests.
    4. Save preprocessed ensemble reference file.

    '''

    print("Merging files from MRI reference and Cognitive Tests reference...")
    df_cog = pd.read_csv(preprocessed_cognitive_data_path).dropna().query("IMAGEUID != 999999 and DIAGNOSIS in @classes")
    df_mri = pd.read_csv(preprocessed_mri_raw_data_path)
    df_mri.rename(columns={'IMAGE_DATA_ID':'IMAGEUID'},inplace=True)
    df_mri['IMAGEUID'] = df_mri['IMAGEUID'].str.replace('I','').astype(np.int64)

    if check_duplicates(df_mri):
        df_mri = remove_duplicates(df_mri)
        print("Duplicates found and removed!")

    print("Normalizing classes from both files...")
    df_ensemble = pd.merge(df_cog,df_mri[['SUBJECT','IMAGEUID','GROUP','MACRO_GROUP','VISIT','ACQ_DATE']],on=['SUBJECT','IMAGEUID'],how=how)
    if 'CN' in df_ensemble['MACRO_GROUP'].unique() or 'AD' in df_ensemble['MACRO_GROUP'].unique() :
        df_ensemble['MACRO_GROUP'].replace({'AD':1,'CN':0,'MCI':2},inplace=True)
    print("Initial data merged!")
    
    df_ensemble = mark_conflicting_diagnosis(df_ensemble)

    print("Saving ensemble reference file...")
    df_ensemble.to_csv(ensemble_data_output_path,index=False)
    return df_ensemble,df_cog,df_mri

def check_duplicates(df_mri): 
    duplicated = df_mri[['SUBJECT','IMAGEUID']].duplicated(keep='first')
    print(f"Found {duplicated.sum()} duplicates!")
    return df_mri[duplicated].shape[0] > 0

def remove_duplicates(df_mri):
    duplicated = df_mri[['SUBJECT','IMAGEUID']].duplicated(keep='first')
    return df_mri[~duplicated]

def mark_conflicting_diagnosis(df_ensemble):
    diff = df_ensemble.query("DIAGNOSIS != MACRO_GROUP")['IMAGEUID']
    print(f"Found {diff.shape[0]} cases with diverging diagnosis between MRI and Cognitive Tests.")
    df_ensemble['CONFLICT_DIAGNOSIS'] = False
    df_ensemble.loc[df_ensemble['DIAGNOSIS'] != df_ensemble['MACRO_GROUP'],'CONFLICT_DIAGNOSIS'] = True
    return df_ensemble

# %%
if __name__ == '__main__':
    preprocessed_cognitive_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv'
    preprocessed_mri_raw_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    classes = [0,1]
    ensemble_data_output_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    df_ensemble,df_cog,df_mri = execute_ensemble_preprocessing(preprocessed_cognitive_data_path,preprocessed_mri_raw_data_path,classes = classes,ensemble_data_output_path=ensemble_data_output_path)

# %%
