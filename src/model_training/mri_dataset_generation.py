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

def generate_mri_dataset_reference(mri_reference_path,
                                output_path,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_sampled_images = 5,
                                sampling_range = 3,
                                num_rotations = 3,
                                save_reference_file = True):

    '''
    Execute MRI metadata preparation for training the deep learning model. The final image will be generated only during training/test/validation step.

    Main Steps:

    - Select orientation of training.
    
    - Select slice indication.

    - Executes Data Augmentation (optional) generating more images based on rotation and flipping. 

    Parameters
    ----------
 
    mri_reference_path: path of the processed MRI reference file.
    
    output_path: path to save the metadata reference file.
    
    orientation: Orientation to slice the image. Values can be "coronal", "sagittal" or "axial".
    
    orientation_slice: Mark to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image.
    
    num_sampled_images: Number of images to sample.
    
    sampling_range: Range to sample new images from original 3D image, with reference to the orientation_slice.

    num_rotations: Number of different rotations to augment original image.

    save_reference_file: Flag to save the reference file or not.
    
    '''

    df_mri_reference = pd.read_csv(mri_reference_path)

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    df_mri_dataset = df_mri_reference.query("ORIENTATION == @orientation and VALID_IMAGE == True")
    df_mri_dataset['MAIN_SLICE'] = orientation_slice
    images= df_mri_reference['IMAGE_DATA_ID'].unique().tolist()
    
    print("Creating augmented samples...")
    df_samples = generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,preprocessed_images = images)
    df_mri_dataset = df_mri_dataset.merge(df_samples,on=['IMAGE_DATA_ID','SLICE'],how='inner').reset_index(drop=True)
    df_mri_dataset['SLICE_ID'] = df_mri_dataset['IMAGE_DATA_ID'] + '_' + df_mri_dataset['SLICE'].astype(str)
    
    if num_rotations > 0:
        print("Creating 2d image rotations...")
        df_samples_rot = generate_augmented_rotations(num_rotations=num_rotations,preprocessed_images=df_mri_dataset['SLICE_ID'])
        df_mri_dataset = df_mri_dataset.merge(df_samples_rot,on='SLICE_ID')

    print("Creating final reference file for prepared images...")
    if save_reference_file:
        now = datetime.now().strftime("%Y%m%d_%H%M")
        reference_file_name = 'PROCESSED_MRI_REFERENCE_' + orientation + '_' + str(orientation_slice) + '_samples_around_slice_' + str(num_sampled_images) +'_num_rotations_' + str(num_rotations) + '_'+ now + '.csv'
        df_mri_dataset.to_csv(output_path+reference_file_name,index=False)
        print("Processed MRI reference file saved at:",output_path+reference_file_name)
    return df_mri_dataset

def generate_augmented_slices(orientation_slice,sampling_range,num_sampled_images,preprocessed_images):
    random.seed(a=None, version=2)
    sampling_population = list(set(range(orientation_slice-sampling_range,orientation_slice+sampling_range+1)) - set([orientation_slice]))
    samples = [(img,random.sample(population= sampling_population,k=num_sampled_images)+[orientation_slice]) for img in preprocessed_images]
    df_samples  = pd.DataFrame(samples,columns=['IMAGE_DATA_ID','SLICE'])
    return df_samples.explode('SLICE').reset_index(drop=True)

def generate_augmented_rotations(num_rotations,preprocessed_images):
    random.seed(a=None, version=2)
    samples = [(img,random.sample(population= list(np.arange(-15,16,2)) ,k=num_rotations) + [0]) for img in preprocessed_images]
    df_samples  = pd.DataFrame(samples,columns=['SLICE_ID','ROTATION_ANGLE'])
    return df_samples.explode('ROTATION_ANGLE').reset_index(drop=True)
