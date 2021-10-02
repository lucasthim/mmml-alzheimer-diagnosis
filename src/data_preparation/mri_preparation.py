# %%
import os
from pathlib import Path
import argparse
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append("./../utils")
from base_mri import *
from utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from mri_augmentation import * 

# %%

def execute_mri_data_preparation(mri_reference_path,
                                ensemble_reference_path,
                                output_path,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_augmented_images = 5,
                                sampling_range = 3,
                                file_format = '.nii.gz'):

    '''
    Execute MRI preparation for training the deep learning model.

    Main Steps:

    - Transform 3D image to 2D image based on an orientation and slice indication

    - Executes Data Augmentation (optional) generating more images based on rotation and flipping. 

    Parameters
    ----------
 
    mri_reference_path: path of the preprocessed MRI reference file.
    
    ensemble_reference_path: Ensemble reference file.

    output_path: path to save the prepared images.
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial".
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image.
    
    num_augmented_images: Number of augmented images to sample.
    
    sampling_range: range to sample new images, with reference to the orientation_slice.
    
    file_format: File format of the (preprocessed) input images.
    
    Example:

        python mri_preparation.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/preprocessed/20210320/" --format ".nii.gz" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/processed/20210327_coronal_50/" --orientation "coronal" --orientation_slice 50 --num_augmented_images 3 --sampling_range 3
    '''

    df_mri_reference = pd.read_csv(mri_reference_path)
    df_ensemble_reference = pd.read_csv(ensemble_reference_path)
    invalid_images = df_ensemble_reference.query("CONFLICT_DIAGNOSIS == True")['IMAGEUID']
    invalid_images = ['I'+str(x) for x in invalid_images]
    images_to_process = df_mri_reference.query("IMAGE_DATA_ID not in @invalid_images")['IMAGE_PATH']

    set_env_variables()
    start = time.time()
    # images_to_process,_,_ = list_available_images(input_path,file_format = file_format)
    print('----------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting data preparation (Cutting 2D Slice + Data Augmentation) for {len(images_to_process)} images. This might take a while... =)")
    print('----------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    for ii,image_path in enumerate(images_to_process):
        if ii == 3: break
        start_img = time.time()
        image_3d = load_mri(path=image_path,as_ants=True)

        print('\n-------------------------------------------------------------------------------------------------------------------')
        if not check_mri_integrity(image_3d):
            print(f"Skipping image ({ii+1}/{len(images_to_process)}) {image_path} because it contains only zeros!")
            continue
            
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)
        print("Transforming 3D MRI to 2D image...")
        
        if num_augmented_images == 0:
            image_2d = slice_image(image_3d,orientation,orientation_slice)
            if image_2d is None:
                print(f"Skipping image ({ii+1}/{len(images_to_process)}) {image_path} because chosen slice contains only zeros!")
                continue
            print("Saving 2d image...")
            save_mri(image=image_2d, output_path = output_path,name=create_file_name_from_path(image_path) + f"_{orientation}_{orientation_slice}",file_format='.npz')

        elif num_augmented_images == 1:
            print("Executing data augmentation on 2d image...")
            augmented_2d_images = generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'simple')
            if augmented_2d_images is None:
                print(f"Skipping image ({ii+1}/{len(images_to_process)}) {image_path} because chosen slice contains only zeros!")
                continue
            print(f"Saving {len(augmented_2d_images.keys())} augmented images...")
            save_batch_mri(image_references=augmented_2d_images, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.npz',verbose=0)
        else:
            print(f"Executing data augmentation for {num_augmented_images} samples within a {sampling_range} voxel distance from the 2d slice {orientation_slice}...")
            augmented_2d_images = generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'neighborhood_sampling')
            if augmented_2d_images is None:
                print(f"Skipping image ({ii+1}/{len(images_to_process)}) {image_path} because chosen slice contains only zeros!")
                continue
            print(f"Saving {len(augmented_2d_images.keys())} augmented images...")
            save_batch_mri(image_references=augmented_2d_images, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.npz',verbose=0)
        
        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)

    print("Creating new reference image table for prepared images...")
    generate_metadata_for_processed_images(output_path,mri_reference_path)
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images prepared! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def generate_metadata_for_processed_images(output_path,mri_reference_path):
    prepared_images,_,_ = list_available_images(output_path,file_format='.npz',verbose=0)
    df_final_reference = create_reference_table(prepared_images,output_path = output_path,previous_reference_file_path=mri_reference_path,save=False)
    df_final_reference.query("IMAGE_PATH == IMAGE_PATH",inplace=True)
    df_final_reference.to_csv(output_path+'REFERENCE.csv',index=False)

# %%

if __name__ == '__main__':
    ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    mri_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    output_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/sample/'
    
    execute_mri_data_preparation(mri_reference_path,
                                ensemble_reference_path,
                                output_path,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_augmented_images = 5,
                                sampling_range = 3,
                                file_format = '.nii.gz')
    generate_metadata_for_processed_images(output_path,mri_reference_path)
# %%

# arg_parser = argparse.ArgumentParser(description='Executes Data Preparation for MR images. Steps include transforming the 3D image into a 2D slice and also a simple Data Augmentation (optional)')

# arg_parser.add_argument('-i','--input',
#                     metavar='input',
#                     type=str,
#                     required=True,
#                     help='Input directory of the mri files')

# arg_parser.add_argument('-f','--format',
#                     metavar='format',
#                     type=str,
#                     default='.nii.gz',
#                     required=False,
#                     help='Format of the mri input files')

# arg_parser.add_argument('-o','--output',
#                     metavar='output',
#                     type=str,
#                     required=True,
#                     help='Output directory of the mri files')

# arg_parser.add_argument('-or','--orientation',
#                     metavar='orientation',
#                     type=str,
#                     default='coronal',
#                     required=True,
#                     help='Orientation to cut the 3D MRI')

# arg_parser.add_argument('-s','--orientation_slice',
#                     metavar='orientation_slice',
#                     type=int,
#                     default=50,
#                     required=True,
#                     help='Slice to cut the 3D MRI')

# arg_parser.add_argument('-a','--num_augmented_images',
#                     metavar='num_augmented_images',
#                     type=int,
#                     default=0,
#                     required=True,
#                     help='Number of augmented images. If equals to zero, no data augmentation is carried')

# arg_parser.add_argument('-r','--sampling_range',
#                     metavar='sampling_range',
#                     type=int,
#                     default=5,
#                     required=False,
#                     help='Range to sample for data augmentation')

# args = arg_parser.parse_args()


# %%
