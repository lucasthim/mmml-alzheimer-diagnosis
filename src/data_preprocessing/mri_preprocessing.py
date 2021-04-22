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

sys.path.append("./../utils")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from base_mri import *
from deepbrain_skull_strip import deep_brain_skull_stripping
from antspy_registration import register_image_with_atlas
from mri_crop import crop_mri_at_center
from mri_standardize import clip_and_normalize_mri
from mri_label import label_image_files

def execute_preprocessing(input_path,output_path,box,skip):
    
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
    
    skip: amount of files to skip when executing preprocessing. This is to be used when reprocessing a batch of files that failed during execution.
    
    Example
    ----------
    
    python mri_preprocessing.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/raw/ADNI" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210402" --skip 0
        
    '''   
    
    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting pre-processing (Labeling + Standardizing + Registration + Skull Stripping + Cropping) for {len(images_to_process)} images. This might take a while... =)")
    print(f"Skipping {skip} images.")
    print('------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)
    
    for ii,image_path in enumerate(images_to_process):
        
        if ii < skip: continue
        
        start_img = time.time()
        input_image = load_mri(path=image_path)
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)

        print("Standardizing image based on Atlas...")
        standardized_image = clip_and_normalize_mri(input_image)

        print("Registering image to Atlas...")
        registered_image:ants.ANTsImage = register_image_with_atlas(standardized_image)
        
        print("Stripping skull from image...")
        stripped_image:ants.ANTsImage = deep_brain_skull_stripping(image=registered_image, probability = 0.5,output_as_array=False)
        
        print("Cropping image with bounding box 100x100x100...")
        cropped_image:ants.ANTsImage = crop_mri_at_center(image=stripped_image,cropping_box=box)

        print("Saving final image...")
        save_mri(image=cropped_image, output_path = output_path,name=create_file_name_from_path(image_path),file_format='.nii.gz')

        total_time_img = (time.time() - start_img)
        print(f'Process for image ({ii+1}/{len(images_to_process)}) took %.2f sec) \n' % total_time_img)
    
    print("Creating new reference image table for preprocessed images...")
    preprocessed_images,_,_ = list_available_images(output_path,file_format='.nii.gz',verbose=0)
    create_images_reference_table(preprocessed_images,output_path = output_path)
    # label_image_files(preprocessed_images,file_format='.nii.gz')
    
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images pre processed! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

# %%
def main():
    execute_preprocessing(input_path=args.input, 
                          output_path=args.output, 
                          box=args.box,
                          skip = args.skip)

arg_parser = argparse.ArgumentParser(description='Preprocess MR images.')

arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=True,
                    help='Input directory of the nifti files.')

arg_parser.add_argument('-o','--output',
                    metavar='output',
                    type=str,
                    required=True,
                    help='Output directory of the nifti files.')

arg_parser.add_argument('-b','--box',
                    dest='box',
                    type=list,
                    default=100,
                    required=False,
                    help='Box to crop brain image.')

arg_parser.add_argument('-s','--skip',
                    dest='skip',
                    type=int,
                    default=0,
                    required=False,
                    help='Amount of images to skip when executing preprocessing.')

args = arg_parser.parse_args()

if __name__ == '__main__':
    main()    
