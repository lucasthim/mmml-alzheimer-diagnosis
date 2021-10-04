import os
import sys
import time
import random
import argparse
from pathlib import Path

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
    
    ensemble_reference_path: Ensemble reference file. Necessary to eliminate conflicting diagnosis cases.

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
    invalid_images = df_ensemble_reference.query("CONFLICT_DIAGNOSIS == True")['IMAGEUID']
    invalid_images = ['I'+str(x) for x in invalid_images]
    df_mri_reference = df_mri_reference.query("IMAGE_DATA_ID not in @invalid_images")
    images_to_process = df_mri_reference['IMAGEUID']

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)
    
    sampled_images = generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,num_preprocessed_images = images_to_process.shape[0])
    df_mri_processed_reference = pd.DataFrame(columns=['IMAGEUID','orientation','orientation_slice','slice_num','rotation_angle'])
    augmented_images_amount = (num_sampled_images + 1) * (num_of_image_rotations + 1)
    new_images_list = int(images_to_process.tolist() * augmented_images_amount)
    df_mri_processed_reference['IMAGEUID'] = new_images_list
    df_mri_processed_reference['orientation'] = orientation
    df_mri_processed_reference['orientation_slice'] = orientation_slice
    df_mri_processed_reference.sort_values(by='IMAGEUID',inplace=True)
    # TODO: unfold sampled sliced and rotations here. Probably use pivot table. Or if it doesnt work, just loop through.
    for ii,image_path in enumerate(images_to_process):
        pass
        augmented_2d_images = generate_augmented_images(orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'neighborhood_sampling')

def generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,num_preprocessed_images):
    random.seed(a=None, version=2)
    sampling_population = list(set(range(orientation_slice-sampling_range,orientation_slice+sampling_range+1)) - set([orientation_slice]))
    samples = [(img,random.sample(population= sampling_population,k=num_sampled_images)+[orientation_slice]) for img in range(num_preprocessed_images)]
    # samples = [item for sublist in samples for item in sublist]
    return samples

def generate_augmented_rotations(num_of_image_rotations,images_to_process):
    random.seed(a=None, version=2)
    samples = [(img,random.sample(population= list(np.arange(-15,16,2)) ,k=num_of_image_rotations) + [0]) for img in images_to_process]
    # samples = [item for sublist in samples for item in sublist]
    return samples

if __name__ == '__main__':
    output_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PROCESSED_MRI_REFERENCE.csv'
    mri_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    
    execute_mri_metadata_preparation(mri_reference_path = mri_reference_path,
                                                                ensemble_reference_path = ensemble_reference_path,
                                                                output_path = output_path)