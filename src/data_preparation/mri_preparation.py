import os
from pathlib import Path
import argparse
import time

import pandas as pd 

import sys
# Linux GPU fix: preload cu12 libcusolver.so.11 before TF imports (no-op off Linux).
# See src/_cuda_preload.py.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
    from _cuda_preload import preload_cusolver
    preload_cusolver()
except Exception:
    pass
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from base_mri import *
from utils.utils import create_file_name_from_path, list_available_images, create_reference_table

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

from mri_augmentation import * 
# IGNORE THIS FILE - see mri_batch_preparation.py instead

def execute_mri_data_preparation(mri_reference_path,
                                output_path,
                                ensemble_reference_path = None,
                                adnimerge_path = None,
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
    
    ensemble_reference_path: Ensemble reference file. Necessary to eliminate conflicting diagnosis cases.

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
    
    if ensemble_reference_path is not None:
        df_ensemble_reference = pd.read_csv(ensemble_reference_path)
        invalid_images = df_ensemble_reference.query("CONFLICT_DIAGNOSIS == True")['IMAGEUID']
        invalid_images = ['I'+str(x) for x in invalid_images]
        images_to_process = df_mri_reference.query("IMAGE_DATA_ID not in @invalid_images")['IMAGE_PATH']
    else:
        images_to_process = df_mri_reference['IMAGE_PATH']
    
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
        # if ii == 3: break
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
    # Merge diagnosis labels from ADNIMERGE (the ruling reference). Prepared-images left
    # join keeps every slice; images absent from ADNIMERGE keep a blank MACRO_GROUP.
    generate_metadata_for_processed_images(output_path, adnimerge_path)
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images prepared! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def generate_metadata_for_processed_images(output_path, adnimerge_path):
    '''
    Build the prepared-images REFERENCE.csv: one row per .npz slice, with the diagnosis
    label merged in from ADNIMERGE (the ruling reference, same as mri_preprocessing.py).

    ADNIMERGE is joined on its integer IMAGEUID against the prepared IMAGE_DATA_ID
    ('I' + IMAGEUID). The join is prepared-images (left) -> ADNIMERGE (right), so EVERY
    prepared slice survives; images absent from ADNIMERGE (e.g. new DICOM scans not yet
    linked) keep a blank MACRO_GROUP rather than being dropped.

    ADNIMERGE's DX is CN / MCI / Dementia; we map Dementia -> AD to match the pipeline's
    CN/AD/MCI label convention (MACRO_GROUP).
    '''
    prepared_images,_,_ = list_available_images(output_path,file_format='.npz',verbose=0)
    df_prepared = create_reference_table(prepared_images, output_path=output_path, save=False)  # paths only, no merge

    if adnimerge_path is not None:
        df_adni = pd.read_csv(adnimerge_path, low_memory=False)
        df_adni = df_adni.dropna(subset=['IMAGEUID', 'DX']).copy()
        df_adni['IMAGE_DATA_ID'] = 'I' + df_adni['IMAGEUID'].astype(int).astype(str)
        df_adni['MACRO_GROUP'] = df_adni['DX'].replace({'Dementia': 'AD'})
        df_labels = df_adni[['IMAGE_DATA_ID', 'MACRO_GROUP']].drop_duplicates('IMAGE_DATA_ID')

        df_final_reference = pd.merge(df_prepared, df_labels, how='left', on='IMAGE_DATA_ID')
        labeled_imgs = df_final_reference[df_final_reference['MACRO_GROUP'].notna()]['IMAGE_DATA_ID'].nunique()
        total_imgs = df_final_reference['IMAGE_DATA_ID'].nunique()
        print(f"Merged ADNIMERGE labels: {labeled_imgs}/{total_imgs} images labeled "
              f"({df_final_reference['MACRO_GROUP'].notna().sum()}/{len(df_final_reference)} slices).")
    else:
        df_final_reference = df_prepared

    df_final_reference.to_csv(output_path+'REFERENCE.csv',index=False)
    print(f"Prepared REFERENCE.csv saved with {len(df_final_reference)} rows -> {output_path}REFERENCE.csv")

if __name__ == '__main__':
    # ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    mri_reference_path = 'data/mri/preprocessed/20260707/REFERENCE.csv'
    adnimerge_path = 'data/tabular/ADNIMERGE.csv'
    output_path = 'data/mri/processed/sample/'

    execute_mri_data_preparation(mri_reference_path=mri_reference_path,
                                # ensemble_reference_path,
                                output_path=output_path,
                                adnimerge_path=adnimerge_path,
                                orientation = 'coronal',
                                orientation_slice = 50,
                                num_augmented_images = 2,
                                sampling_range = 3,
                                file_format = '.nii.gz')
