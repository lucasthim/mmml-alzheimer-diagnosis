# %%
import os
from pathlib import Path
import sys
import argparse
import time
import datetime

import numpy as np
import nibabel as nib
# Linux GPU fix: preload cu12 libcusolver.so.11 before TF imports (no-op off Linux).
# See src/_cuda_preload.py.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
    from _cuda_preload import preload_cusolver
    preload_cusolver()
except Exception:
    pass
# Import TensorFlow/deepbrain before ants (ITK): both ship an OpenMP runtime and, on macOS,
# if ITK's initializes first, TF's session.run deadlocks during skull stripping.
import tensorflow as tf
from deepbrain import Extractor
import ants


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from utils import create_reference_table, list_available_images, create_file_name_from_path
from base_mri import check_mri_integrity, save_mri, load_mri, set_env_variables
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
    arg_parser = argparse.ArgumentParser(
        description='Preprocess raw ADNI MRIs (3D pipeline: standardize -> register to atlas -> skull-strip -> crop 100^3).')

    arg_parser.add_argument('-i', '--input',
                        metavar='input_path',
                        type=str,
                        required=False,
                        default='data/mri/raw/ADNI',
                        help='Folder of raw .nii files, searched recursively (run from the repo root). Default: data/mri/raw/ADNI')

    arg_parser.add_argument('-o', '--output',
                        metavar='output_path',
                        type=str,
                        required=False,
                        default='data/mri/preprocessed/' + datetime.datetime.now().strftime('%Y%m%d'),
                        help='Output folder for preprocessed .nii.gz + REFERENCE.csv. Default: data/mri/preprocessed/<today>')

    arg_parser.add_argument('-b', '--box',
                        metavar='box',
                        type=int,
                        required=False,
                        default=100,
                        help='Center-crop cube size in voxels. Default: 100 (-> 100x100x100).')

    arg_parser.add_argument('-s', '--skip',
                        metavar='skip',
                        type=int,
                        required=False,
                        default=0,
                        help='Skip the first N images (to resume a failed batch).')

    arg_parser.add_argument('-l', '--limit',
                        metavar='limit',
                        type=int,
                        required=False,
                        default=0,
                        help='Process at most N images (0 = all). Use 1 to smoke-test.')

    arg_parser.add_argument('--skip-skull-stripping',
                        action='store_true',
                        help='Bypass DeepBrain skull stripping (registered image goes straight to crop).')

    arg_parser.add_argument('-r', '--mri-reference',
                        metavar='mri_reference_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Optional prior MRI metadata CSV to merge into the output REFERENCE.csv.')

    args = arg_parser.parse_args()

    execute_preprocessing(
        input_path=args.input,
        output_path=args.output,
        box=args.box,
        skip=args.skip,
        limit=args.limit,
        mri_reference_path=args.mri_reference,
        skip_skull_stripping=args.skip_skull_stripping,
    )
