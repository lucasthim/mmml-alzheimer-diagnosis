import os

import ants
import nibabel as nib
import numpy as np

def crop_mri_at_center(image = None, cropping_box = 100,center_dim = None):
    
    '''
    Crop MR image at center of the image. Suggested size = 100x100x100
    
    Params:

    - image: MRI provided in ANTsPyImage format. If provided, function will use it instead of input_path.

    - cropping_box: box size to crop image
    
    - center_dim: custom center point. If not provided, it is calculated the center of the image
    '''

    if image is None:
        image = ants.image_read(input_path)

    lower_dim, upper_dim = get_lower_and_upper_dimensions(image, cropping_box,center_dim)

    if type(image) is np.ndarray:
        return crop_as_numpy(image,lower_dim,upper_dim)
    else:
        return ants.crop_indices(image,lowerind = lower_dim,upperind = upper_dim)


def get_lower_and_upper_dimensions(image, cropping_box, center_dim = None):
    if center_dim is None:
        center_dim = [int(np.ceil(x/2)) for x in image.shape]
    lower_dim = [int(x - cropping_box/2) for x in center_dim]
    upper_dim = [int(x + cropping_box/2) for x in center_dim]
    return lower_dim, upper_dim

def crop_as_numpy(image,lower_dim,upper_dim):
    return image[lower_dim[0]:upper_dim[0],lower_dim[1]:upper_dim[1],lower_dim[2]:upper_dim[2]]