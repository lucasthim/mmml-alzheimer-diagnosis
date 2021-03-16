
import pandas as pd
import numpy as np
import ants

ATLAS_PATH = '/home/lucasthim1/alzheimer_data/Atlas/atlas_t1.nii'
# output_path = "/home/lucasthim1/alzheimer_data/test/registration_test/01_affine_s3/"
# input_path = "/home/lucasthim1/alzheimer_data/test/002_S_4270/MT1__N3m/2011-10-11_07_59_12.0/S125083/ADNI_002_S_4270_MR_MT1__N3m_Br_20111015081648646_S125083_I261073.nii"

def register_image_with_atlas(input_path:str,output_path:str = None,input_img:ants.ANTsImage = None,type_of_transform = 'Affine'):
    
    '''
    Execute ANTs Registration.

    Params:

    - input_path: Path where image to be processed is located.

    - output_path: Path to save the processed image.

    - input_image: Image file in numpy array format. If provided, function will use it instead of input_path

    - type_of_transform: Type of Registration Transformation to be applied. Values tested: 'Affine', 'Similarity', 'Rigid'.
    '''

    if input_img is not None:
        moving = input_img
    elif input_path is not None:
        moving = ants.image_read(input_path)
    else:
        raise("Please specify either an input path or the input image.")

    fixed = ants.image_read(ATLAS_PATH)
    mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform=type_of_transform ,grad_step=0.1)
    warpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    if not output_path:
        return warpedimage
    
    final_img_name = os.path.splitext(os.path.splitext(os.path.basename(input_path))[0])[0]
    output_file_path = output_path + '/' + final_img_name + "_registered_" + type_of_transform + ".nii.gz"
    warpedimage.to_file(filename = output_file_path)

# TODO: maybe put cropping stage here