import os
from pathlib import Path


def list_available_images(input_dir):
    
    '''
    List full path to available images.
    '''

    available_images = []
    print("Looking for MRI raw images in path:",input_dir,'\n')
    
    available_images = list(Path(input_dir).rglob("*.nii"))
    print("Found a total of ",len(available_images)," images.")

    masks_and_wrong_images = list(Path(input_dir).rglob("*[Mm]ask*.nii"))
    print("Found a total of ",len(masks_and_wrong_images)," mask images.")
    
    print("Available images to process: ",len(available_images) - len(masks_and_wrong_images),"\n")
    selected_images = list(set(available_images) - set(masks_and_wrong_images))
    
    return selected_images,available_images,masks_and_wrong_images

def delete_useless_images(input_dir):
    
    available_images = [os.fspath(x) for x in list(Path(input_dir).rglob("*.nii.gz"))]
    if available_images:
        useless_images = [x for x in available_images \
            if "masked_basic" in x \
            or "wm" in x \
            or "gm" in x \
            or "csf" in x \
            or "mask_" in x \
            or "mask." in x ]
        for img in useless_images:
            os.remove(img)

def set_env_variables():
    print("Setting ANTs and NiftyReg environment variables...\n")

    os.environ['ANTSPATH'] = '/home/lucasthim1/ants/ants_install/bin'
    os.environ['PATH'] =os.environ['PATH'] +  ":" + os.environ['ANTSPATH']
    os.environ['NIFTYREG_INSTALL'] = '/home/lucasthim1/niftyreg/niftyreg_install'
    os.environ['PATH'] = os.environ['PATH'] +  ":" + os.environ['NIFTYREG_INSTALL'] + '/bin'