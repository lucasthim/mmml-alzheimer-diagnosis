import os
from pathlib import Path
from typing import Union

import numpy as np
import ants

def list_available_images(input_dir):
    
    '''
    List full path to available images.
    '''

    available_images = []
    print("Looking for MRI raw images in path:",input_dir,'\n')
    
    available_images = list(Path(input_dir).rglob("*.nii"))
    print("Found a total of ",len(available_images)," images.")

    masks_and_wrong_images = list(Path(input_dir).rglob("*[Mm]ask*.nii"))
    print("Found a total of ",len(masks_and_wrong_images)," mask images.")
    
    print("Available images to process: ",len(available_images) - len(masks_and_wrong_images),"\n")
    selected_images = list(set(available_images) - set(masks_and_wrong_images))
    
    return selected_images,available_images,masks_and_wrong_images

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

def create_file_name_from_path(path):
    return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]

def save_mri(image:Union[np.ndarray, ants.ANTsImage],name:str = None,output_path:str = None,file_format:str = '.npz'):
    
    '''

    Save image in memory to a file.

    Parameters:

    - image: image object to save. Can be either a numpy array or an ANTs image.

    - name: name of the file.

    - output_path: directory folder to save the file.

    - file_format: file format of the image. Can be saved as a compressed numpy array (.npz) or a compressed Nifti image (.nii.gz)

    '''

    output_file_path = output_path + '/' + name + file_format
    
    if file_format  == '.npz':
        if type(image) is not np.ndarray: image = image.numpy()
        np.savez_compressed(output_file_path ,image)
        image = ants.from_numpy(image)
    elif file_format == 'nii.gz':
        if type(image) is not ants.ANTsImage: image = ants.from_numpy(image) 
        image.to_file(output_file_path)
    print("Image saved at:",output_file_path)

def load_mri(path:str) -> ants.ANTsImage:
    '''
    Load image from path as an ANTsImage
    '''
    return ants.image_read(path)

def set_env_variables():
    print("Setting ANTs and NiftyReg environment variables...\n")

    os.environ['ANTSPATH'] = '/home/lucasthim1/ants/ants_install/bin'
    os.environ['PATH'] =os.environ['PATH'] +  ":" + os.environ['ANTSPATH']
    os.environ['NIFTYREG_INSTALL'] = '/home/lucasthim1/niftyreg/niftyreg_install'
    os.environ['PATH'] = os.environ['PATH'] +  ":" + os.environ['NIFTYREG_INSTALL'] + '/bin'