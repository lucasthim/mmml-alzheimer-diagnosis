import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from mri_augmentation import *
sys.path.append("./../utils")
from base_mri import *
from utils import *

def execute_mri_metadata_preparation(mri_reference_path,
                                ensemble_reference_path,
                                output_path,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_sampled_images = 5,
                                sampling_range = 3,
                                num_of_image_rotations = 3):

    '''
    Execute MRI metadata preparation for training the deep learning model. The final image will be generated only during training/test/validation step.

    Main Steps:

    - Transform 3D image to 2D image based on an orientation and slice indication

    - Executes Data Augmentation (optional) generating more images based on rotation and flipping. 

    Parameters
    ----------
 
    mri_reference_path: path of the preprocessed MRI reference file.
    
    ensemble_reference_path: Processed ensemble reference file. Necessary to eliminate conflicting diagnosis cases and provide dataset information.

    output_path: path to save the metadata reference file.
    
    orientation: Orientation to slice the image. Values can be "coronal", "sagittal" or "axial".
    
    orientation_slice: Mark to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image.
    
    num_sampled_images: Number of images to sample.
    
    sampling_range: Range to sample new images from original 3D image, with reference to the orientation_slice.

    num_of_image_rotations: Number of different rotations to augment original image.
    
    Example:

        python mri_preparation.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/preprocessed/20210320/" --format ".nii.gz" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/processed/20210327_coronal_50/" --orientation "coronal" --orientation_slice 50 --num_augmented_images 3 --sampling_range 3
    '''
    df_mri_reference = pd.read_csv(mri_reference_path)
    df_ensemble_reference = pd.read_csv(ensemble_reference_path)
    df_ensemble_reference['IMAGE_DATA_ID'] = ['I'+str(x) for x in df_ensemble_reference['IMAGEUID']]
    invalid_images = df_ensemble_reference.query("CONFLICT_DIAGNOSIS == True")['IMAGE_DATA_ID']
    df_mri_reference = df_mri_reference.query("IMAGE_DATA_ID not in @invalid_images")
    images_to_process = df_mri_reference['IMAGE_DATA_ID']

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    print('----------------------------------------------------------------------------------------------------------------------------')
    print(f"Executing MRI Reference preparation (Cutting 2D Slice + Data Augmentation) for {len(images_to_process)} images. This file will be used during CNNs training/test/validation.")
    print('----------------------------------------------------------------------------------------------------------------------------')

    df_mri_processed_reference = pd.DataFrame(columns=['IMAGE_DATA_ID','orientation','orientation_slice','slice_num','IMAGE_SLICE_ID'])

    print("Creating 2d images and samples...")
    df_sampled_images = generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,preprocessed_images = images_to_process)
    df_mri_processed_reference['IMAGE_DATA_ID'] = df_sampled_images['IMAGE_DATA_ID'].copy()
    df_mri_processed_reference['slice_num'] = df_sampled_images['slice_num'].copy()
    df_mri_processed_reference = df_mri_processed_reference.explode('slice_num').reset_index(drop=True)

    df_mri_processed_reference['IMAGE_SLICE_ID'] = df_mri_processed_reference['IMAGE_DATA_ID'] + '_' + df_mri_processed_reference['slice_num'].astype(str)
    print("Creating 2d image rotations...")
    df_rotated_images = generate_augmented_rotations(num_of_image_rotations=3,preprocessed_images=df_mri_processed_reference['IMAGE_SLICE_ID'])
    df_mri_processed_reference = df_mri_processed_reference.merge(df_rotated_images,on='IMAGE_SLICE_ID')
    df_mri_processed_reference = df_mri_processed_reference.explode('rotation_angle').reset_index(drop=True)

    df_mri_processed_reference['orientation'] = orientation
    df_mri_processed_reference['orientation_slice'] = orientation_slice

    print("Separating subjects by dataset (train,validation,test)...")
    df_mri_processed_reference = df_mri_processed_reference.merge(df_mri_reference,on='IMAGE_DATA_ID')
    df_mri_processed_reference = df_mri_processed_reference.merge(df_ensemble_reference[['IMAGE_DATA_ID','DATASET']],on='IMAGE_DATA_ID',how='left')

    now = datetime.now().strftime("%Y%m%d_%H%M")
    reference_file_name = 'PROCESSED_MRI_REFERENCE_'+ now +'_' + orientation + '_' + str(orientation_slice) + '_samples_around_slice_' + str(num_sampled_images) +'_num_rotations_' + str(num_of_image_rotations) + '.csv'
    
    print("Creating final reference file for prepared images...")
    
    df_mri_processed_reference.to_csv(output_path+reference_file_name,index=False)
    print("Processed MRI reference file saved at:",output_path+reference_file_name)
    return output_path+reference_file_name

def generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,preprocessed_images):
    random.seed(a=None, version=2)
    sampling_population = list(set(range(orientation_slice-sampling_range,orientation_slice+sampling_range+1)) - set([orientation_slice]))
    samples = [(img,random.sample(population= sampling_population,k=num_sampled_images)+[orientation_slice]) for img in preprocessed_images]
    df_samples  = pd.DataFrame(samples,columns=['IMAGE_DATA_ID','slice_num'])
    return df_samples

def generate_augmented_rotations(num_of_image_rotations,preprocessed_images):
    random.seed(a=None, version=2)
    samples = [(img,random.sample(population= list(np.arange(-15,16,2)) ,k=num_of_image_rotations) + [0]) for img in preprocessed_images]
    df_samples  = pd.DataFrame(samples,columns=['IMAGE_SLICE_ID','rotation_angle'])
    return df_samples

if __name__ == '__main__':
    # output_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/'
    # mri_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    # ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    output_path = './../../data/'
    mri_reference_path = './../../data/PREPROCESSED_MRI_REFERENCE.csv'
    ensemble_reference_path = './../../data/PROCESSED_ENSEMBLE_REFERENCE.csv'
    execute_mri_metadata_preparation(mri_reference_path = mri_reference_path,
                                                                ensemble_reference_path = ensemble_reference_path,
                                                                output_path = output_path)

