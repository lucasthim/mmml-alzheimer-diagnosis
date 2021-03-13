#%%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys

sys.path.append("utils")
from utils.skull_stripping_ants.s3 import *
#%%

def set_env_variables():
    print("Setting ANTs and NiftyReg environment variables...\n")
    os.environ['ANTSPATH'] = '/home/lucasthim1/ants/ants_install/bin'
    os.environ['PATH'] =os.environ['PATH'] +  ":" + os.environ['ANTSPATH']
    os.environ['NIFTYREG_INSTALL'] = '/home/lucasthim1/niftyreg/niftyreg_install'
    os.environ['PATH'] = os.environ['PATH'] +  ":" + os.environ['NIFTYREG_INSTALL'] + '/bin'


def apply_full_s3_skull_stripping(input_path,output_path):
    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    start = time.time()
    skull_stripper = SkullStripper(input_path, output_path, want_tissues=False, want_atlas=False)
    output_img_path = skull_stripper.strip_skull()
    total_time = (time.time() - start) / 60.
    print('Done with skull stripping! Process took %.2f min) \n' % total_time)

def list_available_images(input_dir):
    
    '''
    List full path to available images
    '''

    available_images = []
    print("Looking for MRI raw images in path:",input_dir,'\n')
    
    available_images = list(Path(input_dir).rglob("*.nii"))
    print("Found a total of ",len(available_images)," images!\n")

    return available_images

def delete_useless_images(input_dir):
    
    available_images = [os.fspath(x) for x in list(Path(input_dir).rglob("*.nii.gz"))]
    if available_images:
        useless_images = [x for x in available_images \
            if "masked_basic" in x \
            or "wm" in x \
            or "gm" in x \
            or "csf" in x \
            or "mask_" in x \
            or "mask." in x ]
        for img in useless_images:
            os.remove(img)

#%%

results = []
if __name__ == '__main__':

    set_env_variables()
    start = time.time()

    input_dir = '/home/lucasthim1/alzheimer_data/raw_mri/ADNI/'
    # TODO: put argument to select between DeepBrain and ANTs
    output_dir = '/home/lucasthim1/alzheimer_data/processed_mri_ants/'

    available_images_dir = list_available_images(input_dir)
    print("Starting ANTs skull stripping process for all images. This might take a while... =)")
    
    for ii,image_path in enumerate(available_images_dir):
        print(f"Processing image ({ii+1}/{len(available_images_dir)}):",image_path)
        apply_full_s3_skull_stripping(image_path,output_dir)
        print("Deleting useless images (masks, gm,wm, etc)...\n")
        delete_useless_images(output_dir)
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('All images processed! Process took %.2f min) \n' % total_time)
    print('-------------------------------------------------------------')

# %%
