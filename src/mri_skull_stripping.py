#%%
import os
from pathlib import Path
import sys
sys.path.append("utils")
import argparse

import numpy as np
import nibabel as nib
from deepbrain import Extractor
from utils.skull_stripping_ants.s3 import *
from deepbrain_skull_strip import *
from base_mri import list_available_images,delete_useless_images,set_env_variables

#%%
arg_parser = argparse.ArgumentParser(description='Executes Skull Stripping to MR images.')

arg_parser.add_argument('-t','--type',
                    dest='type',
                    type=str,
                    default='DeepBrain',
                    help='Skull stripping type. Values: ANTs or DeepBrain')

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
args = arg_parser.parse_args()

#%%

def execute_skull_stripping_process(input_path,output_path,skull_stripping_type = 'ANTs'):
    
    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------')
    print(f"Starting {skull_stripping_type} skull stripping process for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------')
    
    for ii,image_path in enumerate(images_to_process):
        print('\n------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)
        
        if skull_stripping_type == 'ANTs':
            apply_s3_skull_stripping_to_mri(image_path,output_path)
            print("Deleting useless images (masks, gm,wm, etc)...\n")
            delete_useless_images(output_path)
        elif skull_stripping_type == 'DeepBrain':
            apply_deep_brain_skull_stripping_to_mri(image_path.as_posix(),output_path)
        else:
            raise('Please select a valid skull stripping process.')

    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images processed! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def apply_s3_skull_stripping_to_mri(input_path,output_path):
    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    start = time.time()
    skull_stripper = SkullStripper(input_path, output_path, want_tissues=False, want_atlas=False)
    output_img_path = skull_stripper.strip_skull()
    total_time = (time.time() - start) / 60.
    print('Done with skull stripping! Process took %.2f min) \n' % total_time)

def apply_deep_brain_skull_stripping_to_mri(input_path,output_path,probability = 0.5):

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)
    
    start = time.time()
    deep_brain_skull_stripping(input_path = input_path,output_folder = output_path,probability = probability)
    total_time = (time.time() - start)
    print('Done with skull stripping! Process took %.2f sec) \n' % total_time)

#%%

def main():
    '''
    Execute Skull Stripping

    Example:
        python mri_skull_stripping.py --type "DeepBrain" --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI/" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/"

        python mri_skull_stripping.py --type "DeepBrain" --input "/home/lucasthim1/alzheimer_data/test/002_S_4270" --output "/home/lucasthim1/alzheimer_data/test/"
        
        python mri_skull_stripping.py --type "DeepBrain" --input "/home/lucasthim1/alzheimer_data/raw_mri/ADNI" --output "/home/lucasthim1/alzheimer_data/processed_mri_deepbrain/"

    '''    
    
    skull_stripping_type = args.type
    input_path = args.input
    output_path = args.output
    # print(skull_stripping_type)
    # print(input_path)
    # print(output_path)
    execute_skull_stripping_process(input_path=input_path,output_path=output_path,skull_stripping_type = skull_stripping_type)

if __name__ == '__main__':
    main()    

# %%
# input_path = '/home/lucasthim1/alzheimer_data/test/002_S_4270'
# output_path = '/home/lucasthim1/alzheimer_data/test/'
# execute_skull_stripping_process(input_path=input_path,output_path=output_path,skull_stripping_type = 'ANTs')

#%%

