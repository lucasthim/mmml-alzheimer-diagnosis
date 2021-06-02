import os
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import ants

def save_batch_mri(image_references:Union[np.ndarray, ants.ANTsImage],name:str = None,output_path:str = None,file_format:str = '.npz'):

    '''

    Save a batch of MRIs in memory to files.

    Parameters
    ----------

    image_references: Dictionary containing the images and their reference names.

    name: name of the main file.

    output_path: directory folder to save the files.

    file_format: file format of the image. Can be saved as a compressed numpy array (.npz) or a compressed Nifti image (.nii.gz)

    '''

    for key,img in image_references.items():
        mri_name = name + '_' + key 
        save_mri(image = img,name = mri_name,output_path=output_path,file_format=file_format)
 
def save_mri(image:Union[np.ndarray, ants.ANTsImage],name:str = None,output_path:str = None,file_format:str = '.npz'):
    
    '''

    Save image in memory to a file.

    Parameters
    ----------
    
    image: image object to save. Can be either a numpy array or an ANTs image.

    name: name of the file.

    output_path: directory folder to save the file.

    file_format: file format of the image. Can be saved as a compressed numpy array (.npz) or a compressed Nifti image (.nii.gz)

    '''

    output_file_path = output_path + '/' + name + file_format
    
    if file_format  == '.npz':
        if type(image) is not np.ndarray: image = image.numpy()
        np.savez_compressed(output_file_path ,image)
        # image = ants.from_numpy(image)
    elif file_format == '.nii.gz':
        if type(image) is not ants.ANTsImage: image = ants.from_numpy(image) 
        image.to_file(output_file_path)
    print("Image saved at:",output_file_path)

def load_mri(path:str,as_ants=False):
    '''
    Load image from path as an ANTsImage or numpy compressed array.
    '''
    
    if path.endswith(".npz"):
        img= np.load(path)['arr_0']
        if as_ants: img = ants.from_numpy(img)
        return img
    return ants.image_read(path)

def set_env_variables():
    print("Setting ANTs and NiftyReg environment variables...\n")

    os.environ['ANTSPATH'] = '/home/lucasthim1/ants/ants_install/bin'
    os.environ['PATH'] =os.environ['PATH'] +  ":" + os.environ['ANTSPATH']
    os.environ['NIFTYREG_INSTALL'] = '/home/lucasthim1/niftyreg/niftyreg_install'
    os.environ['PATH'] = os.environ['PATH'] +  ":" + os.environ['NIFTYREG_INSTALL'] + '/bin'


def check_mri_integrity(image:Union[np.ndarray, ants.ANTsImage]) -> bool:
    
    if type(image) is not np.ndarray: image = image.numpy()
    
    return image.sum().sum().sum() > 0    


