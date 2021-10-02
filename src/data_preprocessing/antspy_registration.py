
import pandas as pd
import numpy as np
import ants

ATLAS_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/atlas/atlas_t1.nii'
# output_path = "/home/lucasthim1/alzheimer_data/test/registration_test/01_affine_s3/"
# input_path = "/home/lucasthim1/alzheimer_data/test/002_S_4270/MT1__N3m/2011-10-11_07_59_12.0/S125083/ADNI_002_S_4270_MR_MT1__N3m_Br_20111015081648646_S125083_I261073.nii"

def register_image_with_atlas(moving:ants.ANTsImage = None, type_of_transform = 'Affine')-> ants.ANTsImage:
    
    '''
    Execute ANTs Registration.

    Parameters
    ----------

    image: Image file in numpy array format. If provided, function will use it instead of input_path

    type_of_transform: Type of Registration Transformation to be applied. Values tested: 'Affine', 'Similarity', 'Rigid'.
    
    
    Returns
    ----------
    
    Registered image in the ANTsImage format.
    '''

    fixed = ants.image_read(ATLAS_PATH)
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform=type_of_transform ,grad_step=0.1)
    warpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    return warpedimage