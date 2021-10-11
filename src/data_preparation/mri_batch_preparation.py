import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from mri_augmentation import *
sys.path.append("./../utils")
from base_mri import *
from utils import *

def execute_mri_batch_preparation(mri_reference_path,
                                ensemble_reference_path,
                                output_path = '/data/mri/processed/storage/',
                                orientations = {
                                    'coronal':range(35,66),
                                    'axial':range(15,36),
                                    'sagittal':range(15,36),
                                    'axial':range(65,86),
                                    'sagittal':range(65,86)
                                }):

    '''
    Execute MRI preparation for large batches of images for training the deep learning model. The final images are saved in separate folders per subject/image_data_id.

    Main Steps:

    - Transform 3D image to 2D image based on an orientation and slice indication

    - Executes Data Augmentation (optional) generating more images based on rotation and flipping. 

    Parameters
    ----------
 
    mri_reference_path: path of the preprocessed MRI reference file.
    
    ensemble_reference_path: Processed ensemble reference file. Necessary to eliminate conflicting diagnosis cases and provide dataset information.

    output_path: path to save the metadata reference file.
    
    orientations: dict containing orientations and range of slices to generate.
    
    Example:

        python mri_preparation.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/preprocessed/20210320/" --format ".nii.gz" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/processed/20210327_coronal_50/" --orientation "coronal" --orientation_slice 50 --num_augmented_images 3 --sampling_range 3
    '''

    df_mri_reference = pd.read_csv(mri_reference_path)
    df_ensemble_reference = pd.read_csv(ensemble_reference_path)
    df_ensemble_reference['IMAGE_DATA_ID'] = ['I'+str(x) for x in df_ensemble_reference['IMAGEUID']]
    invalid_images = df_ensemble_reference.query("CONFLICT_DIAGNOSIS == True")['IMAGE_DATA_ID']
    df_mri_reference = df_mri_reference.query("IMAGE_DATA_ID not in @invalid_images")
    images_to_process = df_mri_reference['IMAGE_DATA_ID']

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    print('----------------------------------------------------------------------------------------------------------------------------')
    print(f"Executing MRI Batch preparation (Cutting 2D Slices) for {len(images_to_process)} images.")
    print('----------------------------------------------------------------------------------------------------------------------------')


    print("Creating 2d image reference...")
    slices_list = []
    for image_index in df_mri_reference.index:
        sample = df_mri_reference.iloc[image_index]
        image_id = sample['IMAGE_DATA_ID']
        image_path = sample['IMAGE_PATH']
        for orientation,slices in orientations.items():
            print(f"Generating 2D {orientation} slices for image {image_id}...")
            slice_objects = generate_slices(image_id,image_path,orientation,slices)
            
            print(f"Saving {orientation} slices for image {image_id}...")
            slice_objects = save_slices(slice_objects, output_path)
            
            slices_list.extend(slice_objects)
    
    df_mri_processed_reference = pd.DataFrame(slices_list)

    print("Separating subjects by dataset (train,validation,test)...")
    df_mri_processed_reference = df_mri_processed_reference.merge(df_ensemble_reference[['IMAGE_DATA_ID','DATASET']],on='IMAGE_DATA_ID',how='left')

    now = datetime.now().strftime("%Y%m%d_%H%M")
    reference_file_name = 'PROCESSED_MRI_REFERENCE_'+ now + '.csv'
    print("Creating final reference file for prepared images...")
    
    df_mri_processed_reference.to_csv(output_path+reference_file_name,index=False)
    print("Processed MRI reference file saved at:",output_path+reference_file_name)
    
    return output_path+reference_file_name

def generate_slices(image_id,image_path,orientation,slice_indices):
    '''
    
    Slice a 3D image and create a batch of 2D images.
        
    Axis orientation:
    0 - Sagittal
    1 - Coronal
    2 - Axial
    
    Parameters
    ----------
    
    image_id: Image data unique ID.

    image_path: path where preprocessed 3D image is located.
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    
    slices: Points to slice the 3D image. Values range from 0 to 100.

    Returns
    ----------
    Returns a list of 2D images with some metadata.
    
    '''
    image_3d = load_mri(path=image_path,as_ants=True).numpy()
    
    if orientation == 'sagittal' or orientation == 'sagital':
        return slice_sagittal(image_id, image_path, orientation, slice_indices, image_3d)
    
    elif orientation == 'coronal':
        return slice_coronal(image_id, image_path, orientation, slice_indices, image_3d)

    elif orientation == 'axial':
        return slice_axial(image_id, image_path, orientation, slice_indices, image_3d)
    else:
        raise ("Invalid orientation option. Choose one from: sagittal,coronal,axial.")

def slice_axial(image_id, image_path, orientation, slice_indices, image_3d, slice_objects):
    '''
    Return 2d slices for axial orientation with some metadata info.

    PS: Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
    
    '''
    rot = np.rot90(image_3d, k=3, axes=(0,1)).copy()        
    return  [{
                'IMAGE_DATA_ID':image_id,
                'ORIENTATION': orientation,
                'SLICE':i,
                'SLICE_DATA':rot[:,:,i],
                'ORIGINAL_IMAGE_PATH':image_path
            } for i in slice_indices]

def slice_coronal(image_id, image_path, orientation, slice_indices, image_3d):
    '''
    Return 2d slices for coronal orientation with some metadata info.

    PS: Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
    
    '''
    rot = np.rot90(image_3d, k=3, axes=(0,2)).copy()
    return  [{
                'IMAGE_DATA_ID':image_id,
                'ORIENTATION': orientation,
                'SLICE':i,
                'SLICE_DATA':rot[:,i,:],
                'ORIGINAL_IMAGE_PATH':image_path
            } for i in slice_indices]

def slice_sagittal(image_id, image_path, orientation, slice_indices, image_3d):
    '''
    Return 2d slices for sagittal orientation with some metadata info.

    PS: Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
    
    '''
    rot = np.rot90(image_3d, k=3, axes=(1,2)).copy()
    rot = np.rot90(rot, k=2, axes=(0,2)).copy()
    return  [{
                'IMAGE_DATA_ID':image_id,
                'ORIENTATION': orientation,
                'SLICE':i,
                'SLICE_DATA':rot[i,:,:],
                'ORIGINAL_IMAGE_PATH':image_path
            } for i in slice_indices]

def save_slices(slices, output_path):
    
    '''
    Saves a batch of 2d slices.

    Rule for saved data path: <output_path>/<IMAGE_DATA_ID>/<orientation>_<2d slice_number>.npz 
                                            Example: /data/storage/I124661/coronal_50.npz

    '''
    if not output_path.endswith('/'): output_path = output_path + '/'
    output = output_path + slices[0]['IMAGE_DATA_ID']
    for slice in slices:
        slice_num = str(slice['SLICE']) if slice['SLICE'] < 10 else '0'+str(slice['SLICE'])
        mri_name = slice['ORIENTATION'] + '_' + slice_num
        save_mri(image = slice['SLICE_DATA'],name = mri_name,output_path=output,file_format='.npz',verbose=0)
        del slice['SLICE_DATA']

def generate_augmented_rotations(num_of_image_rotations,preprocessed_images):
    random.seed(a=None, version=2)
    samples = [(img,random.sample(population= list(np.arange(-15,16,2)) ,k=num_of_image_rotations) + [0]) for img in preprocessed_images]
    df_samples  = pd.DataFrame(samples,columns=['IMAGE_SLICE_ID','rotation_angle'])
    return df_samples

# if __name__ == '__main__':
    # output_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/'
    # mri_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PREPROCESSED_MRI_REFERENCE.csv'
    # ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    # output_path = './../../data/'
    # mri_reference_path = './../../data/PREPROCESSED_MRI_REFERENCE.csv'
    # ensemble_reference_path = './../../data/PROCESSED_ENSEMBLE_REFERENCE.csv'
    # df = execute_mri_metadata_preparation(mri_reference_path = mri_reference_path,
    #                                                             ensemble_reference_path = ensemble_reference_path,
    #                                                             output_path = output_path)
