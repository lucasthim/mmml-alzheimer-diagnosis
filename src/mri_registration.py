#%%
import os
import sys
import argparse
sys.path.append("utils")

import pandas as pd
import numpy as np
import nibabel as nib
import ants



output_path = "/home/lucasthim1/alzheimer_data/test/registration_test/01_affine_s3/"
input_path = "/home/lucasthim1/alzheimer_data/test/002_S_4270/MT1__N3m/2011-10-11_07_59_12.0/S125083/ADNI_002_S_4270_MR_MT1__N3m_Br_20111015081648646_S125083_I261073.nii"



#%%
arg_parser = argparse.ArgumentParser(description='Executes Affine Registration to MR images.')

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


def execute_skull_stripping_process(input_path,output_path):
    
    set_env_variables()
    start = time.time()

    images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------')
    print(f"Starting Affine Registration process for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------')
    
    for ii,image_path in enumerate(images_to_process):
        print('\n------------------------------------------------------------------------------------')
        print(f"Processing image ({ii+1}/{len(images_to_process)}):",image_path)
        
        register_image_with_atlas(image_path,output_path)

    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images processed! Process took %.2f min) \n' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def register_image_with_atlas(image_path,output_path):
    pass

