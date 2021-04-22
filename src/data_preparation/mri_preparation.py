import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor

# TODO: fix paths to remove sys.path.append
sys.path.append("./../utils")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from base_mri import list_available_images, delete_useless_images, set_env_variables, load_mri, save_mri,save_batch_mri, create_file_name_from_path
from mri_augmentation import * 

arg_parser = argparse.ArgumentParser(description='Executes Data Preparation for MR images. Steps include transforming the 3D image into a 2D slice and also a simple Data Augmentation (optional)')

arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=True,
                    help='Input directory of the mri files')

arg_parser.add_argument('-f','--format',
                    metavar='format',
                    type=str,
                    default='.nii.gz',
                    required=False,
                    help='Format of the mri input files')

arg_parser.add_argument('-o','--output',
                    metavar='output',
                    type=str,
                    required=True,
                    help='Output directory of the mri files')

arg_parser.add_argument('-or','--orientation',
                    metavar='orientation',
                    type=str,
                    default='coronal',
                    required=True,
                    help='Orientation to cut the 3D MRI')

arg_parser.add_argument('-s','--orientation_slice',
                    metavar='orientation_slice',
                    type=int,
                    default=50,
                    required=True,
                    help='Slice to cut the 3D MRI')

arg_parser.add_argument('-a','--num_augmented_images',
                    metavar='num_augmented_images',
                    type=int,
                    default=0,
                    required=True,
                    help='Number of augmented images. If equals to zero, no data augmentation is carried')

arg_parser.add_argument('-r','--sampling_range',
                    metavar='sampling_range',
                    type=int,
                    default=5,
                    required=False,
                    help='Range to sample for data augmentation')

args = arg_parser.parse_args()


def execute_data_preparation(input_path,output_path,orientation,orientation_slice,num_augmented_images,sampling_range,file_format):

    '''
    Execute MRI preparation for training the deep learning model.

    Main Steps:

    - Transform 3D image to 2D image based on an orientation and slice indication

    - Executes Data Augmentation (optional) generating more images based on rotation and flipping. 

    Parameters
    ----------
 
    input_path: path of the preprocessed images to prepare.
    
    output_path: path to save the prepared images.
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial".
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image.
    
    num_augmented_images: Number of augmented images to sample.
    
    sampling_range: range to sample new images, with reference to the orientation_slice.
    
    file_format: File format of the (preprocessed) input images.
    
    Example:

        python mri_preparation.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/preprocessed/20210320/" --format ".nii.gz" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/processed/20210327_coronal_50/" --orientation "coronal" --orientation_slice 50 --num_augmented_images 3 --sampling_range 3
    '''

    set_env_variables()
    start = time.time()
    images_to_process,_,_ = list_available_images(input_path,file_format = file_format)
    print('----------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting data preparation (Cutting 2D Slice + Data Augmentation) for {len(images_to_process)} images. This might take a while... =)")
    print('----------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    for ii,image_path in enumerate(images_to_process):
        
        start_img = time.time()
        image_3d = load_mri(path=image_path,as_ants=True)
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)
        print("Transforming 3D MRI to 2D image...")
        
        if num_augmented_images == 0:
            image_2d = slice_image(image_3d,orientation,orientation_slice)
            print("Saving 2d image...")
            save_mri(image=image_2d, output_path = output_path,name=create_file_name_from_path(image_path) + f"_{orientation}_{orientation_slice}",file_format='.npz')

        elif num_augmented_images == 1:
            print("Executing data augmentation on 2d image...")
            augmented_2d_images = generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'simple')
            print("Saving augmented images...")
            save_batch_mri(image_references=augmented_2d_images, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.npz')
            
        else:
            print(f"Executing data augmentation for {num_augmented_images} samples within a {sampling_range} voxel distance from the 2d slice {orientation_slice}...")
            augmented_2d_images = generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'neighborhood_sampling')
            print("Saving augmented images...")
            save_batch_mri(image_references=augmented_2d_images, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.npz')
            
        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)

    # TODO: create final mri reference dataframe.
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images prepared! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def main():
    execute_data_preparation(args.input,args.output,args.orientation,args.orientation_slice,args.num_augmented_images,args.sampling_range,args.format)

if __name__ == '__main__':
    main()    
