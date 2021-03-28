import os
from pathlib import Path
import sys
import argparse
import time

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor

sys.path.append("./../utils")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from deepbrain_skull_strip import deep_brain_skull_stripping
from base_mri import list_available_images, delete_useless_images, set_env_variables, load_mri, save_mri, create_file_name_from_path
from antspy_registration import register_image_with_atlas
from mri_crop import crop_mri_at_center
from mri_standardize import clip_and_normalize_mri

def execute_preprocessing(input_path,output_path):
    
    '''
    MRI Preprocessing pipeline. 
    
    Main steps:
    
    - MRI standardization
    
    - MRI Registration
    
    - MRI Skull Stripping
    
    - MRI Cropping at 100x100x100

    Example:

        python mri_preprocessing.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/raw/ADNI" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/preprocessed/20210328"
    '''   
    
    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting pre-processing (Standardizing + Registration + Skull Stripping + Cropping) for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    for ii,image_path in enumerate(images_to_process):
        
        start_img = time.time()
        input_image = load_mri(path=image_path.as_posix())
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)

        print("Standardizing image based on Atlas...")
        standardized_image = clip_and_normalize_mri(input_image)

        print("Registering image to Atlas...")
        registered_image:ants.ANTsImage = register_image_with_atlas(standardized_image)
        
        print("Stripping skull from image...")
        stripped_image:ants.ANTsImage = deep_brain_skull_stripping(image=registered_image, probability = 0.5,output_as_array=False)
        
        print("Cropping image with bounding box 100x100x100...")
        cropped_image:ants.ANTsImage = crop_mri_at_center(image=stripped_image)

        print("Saving final image...")
        save_mri(image=cropped_image, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.nii.gz')

        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)

    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images pre processed! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')


def main():
     
    
    execute_preprocessing(input_path=args.input,output_path=args.output)


arg_parser = argparse.ArgumentParser(description='Preprocess MR images.')


arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=True,
                    help='Input directory of the nifti files')

arg_parser.add_argument('-o','--output',
                    metavar='output',
                    type=str,
                    required=True,
                    help='Output directory of the nifti files')

arg_parser.add_argument('-c','--bbox',
                    dest='bbox',
                    type=list,
                    default=[100,100,100],
                    required=False,
                    help='Bounding box to crop brain image')

args = arg_parser.parse_args()


if __name__ == '__main__':
    main()    
