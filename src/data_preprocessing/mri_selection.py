import pandas as pd
import numpy as np

def select_mris_to_download(cognitive_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',classes = [0,1],chunks = 1000,already_downloaded_reference_path = None):
    '''

    Select main MRI files to download based on cognitive tests dataset. These MRIs are already aligned with subject diagnosis data (demographics and cognitive tests). 

    '''
    
    df_cog = pd.read_csv(cognitive_data_path).dropna().query("IMAGEUID != 999999 and DIAGNOSIS in @classes")
    
    if already_downloaded_reference_path is not None:
        df_mri = pd.read_csv(already_downloaded_reference_path)
        df_mri.rename(columns={'IMAGE_DATA_ID':'IMAGEUID'},inplace=True)
        df_mri['IMAGEUID'] = df_mri['IMAGEUID'].str.replace('I','').astype(np.int64)
        df_cog2 = pd.merge(df_cog,df_mri[['SUBJECT','IMAGEUID','GROUP','MACRO_GROUP','ACQ_DATE']],on=['SUBJECT','IMAGEUID'],how='inner')
        images_donwloaded = df_cog2['IMAGEUID'].tolist()
        df_cog = df_cog.query("IMAGEUID not in @images_donwloaded")

    imgs = df_cog['IMAGEUID'].unique().shape[0]
    max_count =  len(range(0,df_cog['IMAGEUID'].unique().shape[0],chunks))
    for count,i in enumerate(range(0,imgs,chunks)):
        print(f"Images to download: {count+1}/{max_count}")
        image_chunk = df_cog.iloc[i:i+chunks]['IMAGEUID'].unique().tolist()
        print("Chunk size:",len(image_chunk))
        print(image_chunk)
        print("-------------------------------")
    df_cog[['IMAGEUID']].to_csv(cognitive_data_path.replace('COGNITIVE_DATA_PREPROCESSED','SELECTED_IMAGES_REFERENCE'),index=False)
    return df_cog


if __name__ == '__main__':
    cognitive_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv'
    mri_raw_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/RAW_MRI_REFERENCE.csv'
    classes = [1,0]
    chunks = 600 
    select_mris_to_download(cognitive_data_path = cognitive_data_path,classes = classes,chunks = chunks,already_downloaded_reference_path = mri_raw_data_path)


    