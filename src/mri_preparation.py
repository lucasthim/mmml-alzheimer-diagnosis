import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor

sys.path.append("utils")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from deepbrain_skull_strip import deep_brain_skull_stripping
from base_mri import list_available_images, delete_useless_images, set_env_variables, load_mri, save_mri, create_file_name_from_path
from antspy_registration import register_image_with_atlas
from crop_mri import crop_mri_at_center
from standardize_mri import clip_and_normalize_mri

arg_parser = argparse.ArgumentParser(description='Executes Data Preparation for MR images. Steps include to transform the 3D image into a 2D slice and also execute a simple Data Augmentation (optional)')

arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=True,
                    help='Input directory of the mri files')

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

arg_parser.add_argument('-s','--slice',
                    metavar='orientation_slice',
                    type=int,
                    default=50,
                    required=True,
                    help='Slice to cut the 3D MRI')

arg_parser.add_argument('-a','--augmented',
                    metavar='num_augmented_images',
                    type=int,
                    default=True,
                    required=True,
                    help='Number of augmented images. If equals to zero, no data augmentation is carried')

arg_parser.add_argument('-r','--range',
                    metavar='sampling_range',
                    type=int,
                    default=5,
                    required=False,
                    help='Range to sample for data augmentation')

args = arg_parser.parse_args()


def execute_data_preparation(input_path,output_path,orientation,orientation_slice,num_augmented_images,sampling_range):

    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('----------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting data preparation (Cutting 2D Slice + Data Augmentation) for {len(images_to_process)} images. This might take a while... =)")
    print('----------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    for ii,image_path in enumerate(images_to_process):
        
        start_img = time.time()
        image_3d = load_mri(path=image_path.as_posix(),as_ants=False)
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)

        print("Transforming to 2D image...")
        image_2d = slice_image(image_3d,orientation,orientation_slice)
        
        if num_augmented_images:
            print("Executing data augmentation...")
            augmented_2d_images = generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range)
            print("Saving augmented images...")
            save_mri(image=augmented_2d_images, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.npz')
        else:
            print("Saving 2d image...")
            save_mri(image=image_2d, output_path = output_path,name=create_file_name_from_path(image_path) + f"_{orientation}_{orientation_slice}",file_format='.npz')

        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)


    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images prepared! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def main():
    
    '''
    Execute Skull Stripping

    Example:
        python mri_preparation.py --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI/" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/"

        python mri_preparation.py --input "/home/lucasthim1/alzheimer_data/test/002_S_4270" --output "/home/lucasthim1/alzheimer_data/test/"

        python mri_preparation.py --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/" --orientation "coronal" --slice 30 --range 5

    '''
    
    execute_preprocessing(input_path=args.input,output_path=args.output,args.orientation,args.orientation_slice,args.data_aug,args.sampling_range)

if __name__ == '__main__':
    main()    
