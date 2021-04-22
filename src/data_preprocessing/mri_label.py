import os
import sys
from pathlib import Path
from typing import Union
import warnings

import ants
import numpy as np
import pandas as pd

sys.path.append("./../utils")
from base_mri import list_available_images, load_reference_table


def label_image_files(image_list,file_format,reference_file_path = "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/reference/MRI_MPRAGE.csv") -> None:
    
    '''
    Rename images to give them labels according to the Alzheimer's Disease stage.
    
    
    Parameters
    ----------
    
    image_list: List containing the path and file name for the images to rename.
    
    file_format: File format of the images.
    
    reference_file_path: Path to the reference file containing information about patients and their disease stage.
    
    Returns
    ----------

    Rename images and saves them in their current paths.

    '''
         
    df = load_reference_table(path = reference_file_path)
    image_dict = create_image_dict_by_id(image_list,file_format)
    df_filtered = filter_reference_table_by_images(df, filter_images=image_dict, file_format = file_format)
    rename_images_with_label(df_filtered)
    print("All images renamed with their respective labels!")


def create_image_dict_by_id(imgs_list,file_format):
    img_dict = {}
    for path in imgs_list: 
        img_dict.update({get_image_id(path,file_format):path})
    return img_dict


def get_image_id(path,file_format):
    return path.split('_')[-1].replace(file_format,'')


def filter_reference_table_by_images(df,filter_images,file_format='.nii.gz'):
    '''
    
    Filter reference table by a list of MRI images.
    
    Parameters
    ----------
    df: Dataframe containing additional information about patients and their MRI scans.
    
    filter_images: dict with images to filter de reference table.
    
    file_format: format of the MRI files.
    
    '''

    df_filtered = df.query("IMAGE_DATA_ID in @filter_images.keys()").sort_values("IMAGE_DATA_ID")
    if df_filtered.shape[0] == 0: 
        raise("Images passed are not contained in the reference dataframe.")
    
    df_filtered['IMAGE_DATA_NAME'] = np.sort(list(filter_images.values()))
    df_filtered['LABELED_IMAGE_DATA_NAME'] = df_filtered['IMAGE_DATA_NAME'].str.replace("ADNI_","").str.replace(file_format,"") + '_label-' + df_filtered['MACRO_GROUP'] + '-' + file_format
    
    # if df_filtered.shape[0] != len(filter_images.keys()):
    #     warnings.warn("Not all images passed were found in the reference dataframe.")
    #     print("Total images passed:",len(filter_images.keys()))
    #     print("Total images referenced:",df_filtered.shape[0])
    return df_filtered


def rename_images_with_label(df_reference):
    for old_name,new_name in zip(df_reference['IMAGE_DATA_NAME'],df_reference['LABELED_IMAGE_DATA_NAME']):
        os.rename(old_name,new_name)

def save_reference_file(df_reference,path):
    df_reference.to_csv(path + 'REFERENCE.csv')
        
if __name__ == '__main__':
    
    input_path = "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/preprocessed/small_test/20210320"
    images_to_process,_,_ = list_available_images(input_path,file_format = ".npz")
    label_image_files(images_to_process,file_format = ".npz",reference_file_path = "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/reference/MRI_MPRAGE.csv")