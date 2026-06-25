import pandas as pd
import numpy as np
import argparse

def select_mris_to_download(
    cognitive_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
    classes = [0,1],
    chunks = 1000,
    existing_reference_path = None):
    '''
    Select main MRI files to download based on cognitive tests dataset. 
    
    These MRIs are already aligned with subject diagnosis data (demographics and cognitive tests). 

    PS: After downloading the images, make sure to download the corresponding metadata reference file.
    '''
    
    df_cog = pd.read_csv(cognitive_data_path).query("IMAGEUID != 999999 and DIAGNOSIS in @classes")
    # df_cog = pd.read_csv(cognitive_data_path).dropna().query("IMAGEUID != 999999 and DIAGNOSIS in @classes")  # original: the .dropna() required a COMPLETE cognitive battery incl. MOCA, which ADNI1/GO/2 lack -> it silently kept only ADNI3/4. Removed so all phases are selected.
    
    if existing_reference_path is not None:
        df_cog = filter_images(df_cog, existing_reference_path)

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

def filter_images(df_cog, existing_reference_path):
    "Drop images already present in an existing MRI reference file."
    df_mri = pd.read_csv(existing_reference_path)
    df_mri.rename(columns={'IMAGE_DATA_ID':'IMAGEUID'},inplace=True)
    df_mri['IMAGEUID'] = df_mri['IMAGEUID'].str.replace('I','').astype(np.int64)
    df_cog_merged = pd.merge(df_cog,df_mri[['SUBJECT','IMAGEUID','GROUP','MACRO_GROUP','ACQ_DATE']],on=['SUBJECT','IMAGEUID'],how='inner')
    images_downloaded = df_cog_merged['IMAGEUID'].tolist()
    return df_cog.query("IMAGEUID not in @images_downloaded")


arg_parser = argparse.ArgumentParser(description='Select MRIs to download.')

arg_parser.add_argument('-co','--cognitive',
                    metavar='cognitive_data_path',
                    type=str,
                    required=False,
                    default='data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                    help='Path to the COGNITIVE_DATA_PREPROCESSED.csv file (run from the repo root).')

arg_parser.add_argument('-ch','--chunks',
                    metavar='chunks',
                    type=int,
                    required=False,
                    default=1000,
                    help='Chunk/batch size of image groups to print on console.')

arg_parser.add_argument('-cl','--classes',
                    metavar='classes',
                    type=int,
                    nargs='+',
                    required=False,
                    default=[0,1],
                    help='Alzheimer classes to select, space-separated (e.g. -cl 0 1). CN=0, AD=1, MCI=2.')

arg_parser.add_argument('-e','--existing',
                    metavar='existing_data_path',
                    type=str,
                    required=False,
                    default=None,
                    help='Existing MRI reference directory.')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    select_mris_to_download(
        cognitive_data_path=args.cognitive,
        classes=args.classes,
        chunks=args.chunks,
        existing_reference_path=args.existing,
    )
