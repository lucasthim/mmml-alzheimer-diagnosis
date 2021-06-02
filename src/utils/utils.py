import os
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

def list_available_images(input_dir,file_format = '.nii',verbose=1):

    '''
    List full path to available images.
    
    Params
    ---------------------
    
    input_dir: input directory to read the image files
    
    file_format: file format of the images
    
    
    Returns
    ---------------------
    
    selected_images: Selected images that can be processed
    
    available_images: All the available images in the provided directory
    
    masks_and_wrong_images: Masks and other images that will not be processed
    '''

    available_images = []
    if verbose > 0: print("Looking for MRI images in path:",input_dir,'\n')
    
    available_images = list(Path(input_dir).rglob("*"+file_format))
    if verbose > 0: print("Found a total of ",len(available_images)," images.")

    masks_and_wrong_images = list(Path(input_dir).rglob("*[Mm]ask*"+file_format))
    if verbose > 0: print("Found a total of ",len(masks_and_wrong_images)," mask images.")
    
    if verbose > 0: print("Available images to process: ",len(available_images) - len(masks_and_wrong_images),"\n")
    selected_images = list(set(available_images) - set(masks_and_wrong_images))
    
    if selected_images: 
        selected_images = [x.as_posix() for x in selected_images]

    if available_images: 
        available_images = [x.as_posix() for x in available_images]

    if masks_and_wrong_images:
        masks_and_wrong_images = [x.as_posix() for x in masks_and_wrong_images]
        
    return selected_images,available_images,masks_and_wrong_images

def delete_useless_images(input_dir,file_format = ".nii.gz"):
    
    available_images = [os.fspath(x) for x in list(Path(input_dir).rglob("*"+file_format))]
    if available_images:
        useless_images = [x for x in available_images \
            if "masked_basic" in x \
            or "wm" in x \
            or "gm" in x \
            or "csf" in x \
            or "mask_" in x \
            or "mask." in x ]
        for img in useless_images:
            os.remove(img)

def create_file_name_from_path(path):
    return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]

def load_reference_table(path = "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/reference/MRI_MPRAGE.csv",df = None):
    '''
    
    Loads reference table containing additional information about patients and their MRI scans. 
    
    '''
    
    if df is None:
        df = pd.read_csv(path,engine='python')
    df.columns = [x.replace(' ','_').upper() for x in df.columns]
    
    if 'MACRO_GROUP' not in df.columns:
        df['MACRO_GROUP'] = df['GROUP']
        df.loc[df['MACRO_GROUP'] == 'SMC','MACRO_GROUP'] = 'CN'
        df.loc[df['MACRO_GROUP'] == 'EMCI','MACRO_GROUP'] = 'MCI'
        df.loc[df['MACRO_GROUP'] == 'LMCI','MACRO_GROUP'] = 'MCI'
    
    if 'SUBJECT_IMAGE_ID' not in df.columns:
        df['SUBJECT_IMAGE_ID'] = df['SUBJECT'] + "#" + df['IMAGE_DATA_ID'] 
    return df

def create_reference_table(image_list,output_path,previous_reference_file_path = None) -> None:
    
    '''
    Create a new reference dataframe based on the current list of images.

    Reference dataframe contains information about the patient and their Alzheimer's Disease current situation.
    
    Parameters
    ----------
    
    image_list: List containing the path and file name for the images to rename.
    
    output_path: Path to save the new the reference file. Should be the same path where the images are located
    
    previous_reference_file_path: Path to the previous reference file containing information about patients and their disease stage.
    
    Returns
    ----------

    Saves a reference dataframe along with the images to be further used in the training phase.

    '''

    image_references = create_image_references(image_list)

    df_image_path = pd.DataFrame(data = {
        'SUBJECT_IMAGE_ID':[x[0] for x in image_references], 
        'SUBJECT_ID':[x[1] for x in image_references], 
        'IMAGE_DATA_ID':[x[2] for x in image_references], 
        'IMAGE_PATH':[x[3] for x in image_references]})
    
    if previous_reference_file_path is not None:
        df_original_reference = load_reference_table(previous_reference_file_path)
        df_reference = pd.merge(df_original_reference,df_image_path,how='left',on='IMAGE_DATA_ID')
    else:
        df_reference = df_image_path
    
    if not output_path.endswith('/'): output_path = output_path + '/'
    df_reference['IMAGE_PATH'] = df_reference['IMAGE_PATH'].str.replace(output_path,'')
    df_reference.to_csv(output_path + 'REFERENCE.csv',index=False)
    print("Reference file saved at: ",output_path)
    return df_reference

def create_image_references(imgs_list):
    "Returns a list of image references containing unique_patient_image_id, patient_id, unique_image_id and path to image file, respectively."
    imgs = []
    for path in imgs_list: 

        img_id = 'I'+path.split('_I')[-1].split('_')[0]
        if '.' in img_id:
            img_id =  img_id.split('.')[0]
        unique_image_id = 'I'+path.split('_I')[-1].split('.')[0]
        
        patient_splits = path.split('/')[-1].split('_')
        patient_id = patient_splits[1] + '_' + patient_splits[2] + '_' +patient_splits[3]

        unique_patient_image_id = patient_id + "#" + img_id
        imgs.append((unique_patient_image_id,patient_id, unique_image_id, path))

    return imgs

# def create_image_references(imgs_list):
#     "Creates a triplet where: the first position is subject id, the second is the image id and the third is the path to the image."
#     imgs = []
#     for path in imgs_list: 
#         imgs.append(create_unique_patient_image_id(path))
#     return imgs
