import os

import ants
import nibabel as nib
import numpy as np


def clip_and_normalize_mri(image, lower_bound = 0.02, upper_bound = 99.8)-> ants.ANTsImage:
    
    '''
    
    Execute outlier clipping and image normalization based on the Atlas.

    Params:

    - image: MRI provided in ANTsPyImage format. If provided, function will use it instead of input_path.

    - lower_bound: lower percentile to clip outliers in image. 

    - upper_bound: upper percentile to clip outliers in image.

    '''

    lower_threshold,upper_threshold = get_percentiles(image.numpy(),lower_bound=lower_bound, upper_bound = upper_bound)
    image_clipped = clip_image_intensity(image.numpy(),lower_threshold=lower_threshold, upper_threshold=upper_threshold)
    lower_atlas_threshold, upper_atlas_threshold = get_atlas_thresholds()
    image_scaled = scale_image_linearly(image_clipped,lower_atlas_threshold,upper_atlas_threshold)
    image_scaled = ants.from_numpy(image_scaled, direction=image.direction)

    return image_scaled


def get_percentiles(img,lower_bound=0.02,upper_bound = 99.8):
    img_flatten = img.ravel()
    lower_perc = np.percentile(img_flatten,q=lower_bound)
    upper_perc = np.percentile(img_flatten,q=upper_bound)
    return lower_perc,upper_perc

def scale_image_linearly(img_array:np.ndarray,lower_bound,upper_bound):
    img_array = (img_array - lower_bound) / (upper_bound - lower_bound)
    return img_array

def clip_image_intensity(image:np.ndarray,lower_threshold,upper_threshold):
    image[image > upper_threshold] = upper_threshold
    image[image < lower_threshold] = lower_threshold
    return image

def get_atlas_thresholds(atlas_path = None,lower_bound=0.02,upper_bound=99.8):
    
    if atlas_path is None: return (0.05545412003993988, 92.05744171142578) #for 0.02 and 99.8

    fixed = ants.image_read(atlas_path)
    return get_percentiles(fixed,lower_bound=lower_bound, upper_bound = upper_bound)