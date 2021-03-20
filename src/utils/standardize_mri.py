import os

import ants
import nibabel as nib
import numpy as np

def crop_mri_at_center(input_path:str,output_path:str = None, image = None,bbox_size = 100,center_dim = None):
    '''
    
    Crop MR image at center of the image. Suggested size = 100x100x100
    
    Params:

    - input_path: Path where image to be processed is located.

    - output_path: Path to save the processed image. If not provided, function returns the image object.

    - image: MRI provided in ANTsPyImage format. If provided, function will use it instead of input_path.

    - bbox_size: bounding box size to crop image
    
    - center_dim: custom center point. If not provided, it is calculated the center of the image

    '''

    if image is None:
        image = ants.image_read(input_path)

    if center_dim is None:
        center_dim = [int(np.ceil(x/2)) for x in image.shape]
    lower_dim = [int(x - bbox_size/2) for x in center_dim]
    upper_dim = [int(x + bbox_size/2) for x in center_dim]

    final_img =  ants.crop_indices(image,lowerind = lower_dim,upperind = upper_dim)

    if not output_path: 
        return final_img
    final_img_name = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]
    output_file_path = output_path + '/' + final_img_name + "_masked_deepbrain.nii.gz"
    # final_img_nii = nib.Nifti1Image(final_img, np.eye(4))
    # final_img_nii.header.get_xyzt_units()
    # final_img_nii.to_filename(output_file_path)
    final_img.to_file(output_file_path)

def clip_and_normalize_mri(image, lower_bound = 0.02, upper_bound = 99.8):
    
    '''
    
    Execute outlier clipping and image normalization based on the Atlas.

    - image: MRI provided in ANTsPyImage format. If provided, function will use it instead of input_path.

    - lower_bound: lowerpercentile th
    '''
    
    lower_threshold,upper_threshold = get_percentiles(image,lower_bound=lower_bound, upper_bound = upper_bound)
    image_clipped = clip_image_intensity(image.numpy(),lower_threshold=lower_threshold, upper_threshold=upper_threshold)
    lower_atlas_threshold, upper_atlas_threshold = get_atlas_thresholds()
    image_scaled = scale_image_linearly(image_clipped,lower_atlas_threshold,upper_atlas_threshold)
    return image_scaled


def get_percentiles(img,lower_bound=0.02,upper_bound = 99.8):
    img_flatten = img.numpy().ravel()
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
