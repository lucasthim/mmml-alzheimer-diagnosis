import os
from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor


def deep_brain_skull_stripping(image, probability = 0.5, output_as_array=True):
    
    '''
    Executes Skull Stripping process with the DeepBrain Extraction tool.

    DeepBrain uses a 3D Unet to strip the skulls from patients.

    Params:

    - image: MRI object to strip.

    - probability: Probability to make extraction mask binary and apply to image.

    - output_as_array: Flag to return image as a numpy array and avoid unecessary conversion of objects.
    '''
    
    # execute brain extraction
    ext = Extractor()
    print("Running DeepBrain Skull Stripping...")
    prob = ext.run(image) 
    mask = prob > probability
    
    # apply mask
    final_img = image.copy()
    final_img[~mask] = 0
    print('DeepBrain skull stripping finished.')

    if output_as_array:
        return final_img
    
    return ants.from_numpy(final_img,direction=image.direction)