import os
from pathlib import Path
from typing import Union
import sys

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor

def deep_brain_skull_stripping(image: ants.ANTsImage, probability = 0.5, output_as_array=True,get_mask=False) -> ants.ANTsImage:
    
    '''
    Executes Skull Stripping process with the DeepBrain Extraction tool.

    DeepBrain uses a 3D Unet to strip the skulls from patients.

    Params:

    - image: MRI object to strip.

    - probability: Probability to make extraction mask binary and apply to image.

    - output_as_array: Flag to return image as a numpy array and avoid unecessary conversion of objects.
    
    - image_direction: direction properties from ANTsImage object. This will correctly orient the sagittal, coronal and axias views of the MRI.
    
    - get_mask: Flag to return the skull stripping mask instead of the stripped image.
    
    '''
    
    if type(image) is ants.ANTsImage:
        image_direction = image.direction
        image = image.numpy()
    else:
        image_direction = None
    
    ext = Extractor()
    print("Running DeepBrain Skull Stripping...")
    prob = ext.run(image) 
    mask = prob > probability
    print('DeepBrain skull stripping finished.')
    
    if get_mask:
        mask[mask] = 1
        mask[~mask] = 0
        return mask
    
    # apply mask
    final_img = image.copy()
    final_img[~mask] = 0

    if output_as_array:
        return final_img
    
    if image_direction is not None:
        return ants.from_numpy(final_img,direction=image_direction)
    return ants.from_numpy(final_img)
    