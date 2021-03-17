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
from deepbrain_skull_strip import deep_brain_skull_stripping
from base_mri import list_available_images,delete_useless_images,set_env_variables
from antspy_registration import register_image_with_atlas
from crop_mri import crop_mri_at_center

arg_parser = argparse.ArgumentParser(description='Executes Skull Stripping to MR images.')


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


def execute_preprocessing(input_path,output_path):

    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting pre-processing (Registration + Skull Stripping + Cropping) for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------------------------------------------')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    for ii,image_path in enumerate(images_to_process):
        print('\n-------------------------------------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)
        
        start_img = time.time()

        # TODO: Apply Standardization
        # print("Standardizing image based on Atlas...")
        
        print("Registering image to Atlas...")
        image_to_process = register_image_with_atlas(input_path=image_path.as_posix(),output_path = None,input_img=None)
        
        print("Stripping skull from image...")
        stripped_image = deep_brain_skull_stripping(input_path = input_path, output_path = None, input_img=image_to_process.numpy(), probability = 0.5)
        
        print("Cropping image with bounding box 100x100x100...")
        crop_mri_at_center(input_path = input_path, output_path = output_path, image=stripped_image)

        # TODO: create method to save image and remove saving logic from other methods. method will be flexible and accept ANTsPyImage, numpy and nibabel formats. 
        # It will save image as .nii.gz or as numpy matrix. Whatever is smaller in size 
        # TODO: Create message to print where the image was saved

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
    
    '''
    Execute Skull Stripping

    Example:
        python mri_preprocessing.py --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI/" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/"

        python mri_preprocessing.py --input "/home/lucasthim1/alzheimer_data/test/002_S_4270" --output "/home/lucasthim1/alzheimer_data/test/"

        python mri_preprocessing.py --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/"

    '''    
    
    execute_preprocessing(input_path=args.input,output_path=args.output)

if __name__ == '__main__':
    main()    
