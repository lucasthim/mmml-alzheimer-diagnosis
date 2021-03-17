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
