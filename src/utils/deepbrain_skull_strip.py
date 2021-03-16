import os
from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import ants
from deepbrain import Extractor


def deep_brain_skull_stripping(input_path:str,output_path:str = None,input_img:np.array = None,probability = 0.5):
    '''
    Executes Skull Stripping process with the DeepBrain Extraction tool.

    DeepBrain uses a 3D Unet to strip the skulls from patients.

    Params:

    - input_path: Path where image to be processed is located.

    - output_path: Path to save the processed image.

    - input_image: Image file in numpy array format. If provided, function will use it instead of input_path

    - probability: Probability to make extraction mask binary and apply to image.

    '''

    if input_img is not None:
        img = input_img
    elif input_path is not None:
        img = ants.image_read(input_path).numpy()
    else:
        raise("Please a numpy array as the input image and the input path.")
    
    # execute brain extraction
    ext = Extractor()
    print("Running DeepBrain Skull Stripping...")
    prob = ext.run(img) 
    mask = prob > probability
    
    # apply mask
    final_img = img.copy()
    final_img[~mask] = 0
    print('DeepBrain skull stripping finished.')

    if not output_path: 
        return final_img
    final_img_name = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]
    output_file_path = output_path + '/' + final_img_name + "_masked_deepbrain.nii.gz"
    final_img_nii = nib.Nifti1Image(final_img, np.eye(4))
    final_img_nii.header.get_xyzt_units()
    final_img_nii.to_filename(output_file_path)

    print('Skull stripped image saved as :',output_file_path)
    # return final_img