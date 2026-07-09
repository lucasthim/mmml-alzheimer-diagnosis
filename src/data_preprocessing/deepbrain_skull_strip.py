import os
from pathlib import Path
from typing import Union
import sys

import numpy as np
import nibabel as nib
# TensorFlow (pulled in by deepbrain) must be imported before ants (ITK): both ship an
# OpenMP runtime and, on macOS, if ITK's initializes first, TF's session.run deadlocks.
from deepbrain import Extractor
import ants
try:  # antspyx >=0.4 moved ANTsImage out of the top-level namespace
    from ants.core.ants_image import ANTsImage
except ImportError:  # older antspyx exposed it as ants.ANTsImage
    from ants import ANTsImage

ext = Extractor()

def deep_brain_skull_stripping(image: ANTsImage, probability = 0.5, output_as_array=True,get_mask=False) -> ANTsImage:
    
    '''
    Executes Skull Stripping process with the DeepBrain Extraction tool.

    DeepBrain uses a 3D Unet to strip the skulls from patients.


    Parameters
    ----------

    image: MRI object to strip.

    probability: Probability to make extraction mask binary and apply to image.

    output_as_array: Flag to return image as a numpy array and avoid unecessary conversion of objects.
    
    image_direction: direction properties from ANTsImage object. This will correctly orient the sagittal, coronal and axias views of the MRI.
    
    get_mask: Flag to return the skull stripping mask instead of the stripped image.
        

    Returns
    ----------

    final_img: skull stripped image in ANTsImage format.

    '''

    if isinstance(image, ANTsImage):
        image_direction = image.direction
        image = image.numpy()
    else:
        image_direction = None
    
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

    