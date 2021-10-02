# %%
import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

sys.path.append("./../utils")
from utils import *
from base_mri import *
from deepbrain_skull_strip import deep_brain_skull_stripping
from antspy_registration import register_image_with_atlas
from mri_crop import crop_mri_at_center
from mri_standardize import clip_and_normalize_mri
# from mri_label import label_image_files

def execute_preprocessing(input_path = None,output_path = None,images_to_process = None,box = 100,skip = 0,limit = 0,mri_reference_path = None,skip_skull_stripping=False):
    
    '''
    MRI Preprocessing pipeline. 
    
    Main steps:
    
    - MRI standardization
    
    - MRI Registration
    
    - MRI Skull Stripping
    
    - MRI Cropping at 100x100x100

    Parameters
    ----------
    
    input_path: path where raw MRIs are located.
    
    output_path: path to save preprocessed MRIs.
    
    images_to_process: custom list of images to preprocess.

    skip: amount of files to skip when executing preprocessing. This is to be used when reprocessing a batch of files that failed during execution.
    
    limit: max amount of files to process when executing preprocessing. This is to be used when reprocessing a batch of files that failed during execution.
    
    Example
    ----------
    
    python mri_preprocessing.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/raw/ADNI" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210402" --skip 0
        
    '''   
    
    set_env_variables()
    start = time.time()

    if images_to_process is None:
        images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting pre-processing (Labeling + Standardizing + Registration + Skull Stripping + Cropping) for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------------------------------------------')

    if skip > 0 and limit > 0:
        images_to_process = images_to_process[skip:limit]
        print(f"Processing from  image {skip} to image {limit}.")

    elif skip > 0:
        images_to_process = images_to_process[skip:]
        print(f"Processing from image {skip}.")

    elif limit > 0:
        images_to_process = images_to_process[:limit]
        print(f"Processing up to image {limit}.")
    
    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)
    
    for ii,image_path in enumerate(images_to_process):
        
        start_img = time.time()
        input_image = load_mri(path=image_path)
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)

        print("Standardizing image based on Atlas...")
        standardized_image = clip_and_normalize_mri(input_image)

        print("Registering image to Atlas...")
        registered_image: ants.ANTsImage = register_image_with_atlas(standardized_image)
        
        if not skip_skull_stripping:
            print("Stripping skull from image...")
            stripped_image: ants.ANTsImage = deep_brain_skull_stripping(image=registered_image, probability = 0.5,output_as_array=False)
        else:
            print("Skipping skull stripping step...")
            stripped_image = registered_image

        print("Cropping image with bounding box 100x100x100...")
        cropped_image: ants.ANTsImage = crop_mri_at_center(image=stripped_image,cropping_box=box)

        print("Checking if image is usable...")
        integrity_check = check_mri_integrity(cropped_image)
        if integrity_check:
            print("Saving final image...")
            save_mri(image=cropped_image, output_path = output_path,name= create_file_name_from_path(image_path),file_format='.nii.gz')
        else:
            print("Skipping current image because skull stripping process failed!")
        
        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)
    
    print("Creating new reference image table for preprocessed images...")
    generate_metadata_for_preprocessed_images(output_path,mri_reference_path)
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images pre processed! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def generate_metadata_for_preprocessed_images(output_path,mri_reference_path):
    preprocessed_images,_,_ = list_available_images(output_path,file_format='.nii.gz',verbose=0)
    create_reference_table(preprocessed_images,output_path = output_path,previous_reference_file_path=mri_reference_path)
    # label_image_files(preprocessed_images,file_format='.nii.gz')
    
# %%

if __name__ == '__main__':
    execute_preprocessing(input_path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/raw/ADNI/', 
                        #   output_path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210523/', 
                          output_path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/preprocessed/20211002/', 
                          box=100,
                          skip = 0,
                          limit = 0)

# arg_parser = argparse.ArgumentParser(description='Preprocess MR images.')

# arg_parser.add_argument('-i','--input',
#                     metavar='input',
#                     type=str,
#                     required=False,
#                     default='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/raw/ADNI/',
#                     help='Input directory of the nifti files.')

# arg_parser.add_argument('-o','--output',
#                     metavar='output',
#                     type=str,
#                     required=False,
#                     default='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210523/',
#                     help='Output directory of the nifti files.')

# arg_parser.add_argument('-b','--box',
#                     dest='box',
#                     type=list,
#                     default=100,
#                     required=False,
#                     help='Box to crop brain image.')

# arg_parser.add_argument('-s','--skip',
#                     dest='skip',
#                     type=int,
#                     default=0,
#                     required=False,
#                     help='Amount of images to skip when executing preprocessing.')

# args = arg_parser.parse_args()


# %%

# selected_images,available_images,masks_and_wrong_images = list_available_images(input_dir = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/raw/ADNI/',file_format = '.nii',verbose=1)
