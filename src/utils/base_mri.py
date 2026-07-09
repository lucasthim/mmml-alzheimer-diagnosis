import os
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import ants
# antspyx >=0.4 moved ANTsImage out of the top-level namespace; restore ants.ANTsImage
# so the type annotations below resolve regardless of antspyx version.
if not hasattr(ants, 'ANTsImage'):
    from ants.core.ants_image import ANTsImage as _ANTsImage
    ants.ANTsImage = _ANTsImage

# Reorient DICOM volumes to match the .nii scans (SAR), so registration sees both
# modalities in the same frame. See load_dicom_series.
DICOM_TARGET_ORIENTATION = 'SAR'


def load_dicom_series(path: str, target_orientation: str = DICOM_TARGET_ORIENTATION) -> "ants.ANTsImage":
    '''
    Read a DICOM series folder as a single, spatially-correct 3D ANTsImage.

    Why not ants.dicom_read: it stacks the frames in file/instance order without
    sorting on ImagePositionPatient. ADNI accelerated MPRAGE series store slices out
    of spatial order, so ants.dicom_read produces a shredded/striped volume. We read
    with SimpleITK's ImageSeriesReader, which sorts frames by geometry, hand the
    volume (with its spacing/origin/direction) to ANTs, then reorient to
    target_orientation so DICOM output matches the .nii scans (SAR).
    '''
    import SimpleITK as sitk

    if not os.path.isdir(path):
        path = os.path.dirname(path)  # a .dcm file was passed; use its series folder

    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(path)
    if not series_files:
        raise ValueError(f"No DICOM series found in {path}")
    reader.SetFileNames(series_files)
    sitk_img = reader.Execute()

    # Some ADNI scans are 4D DICOM (a multiframe/enhanced series read as e.g.
    # 256x256x211x1). Drop the trailing singleton so the volume is 3D before we read
    # its geometry (a 4D image also has 4D spacing/direction).
    if sitk_img.GetDimension() == 4:
        size = list(sitk_img.GetSize())
        if size[3] != 1:
            raise ValueError(f"4D DICOM with {size[3]} volumes (expected 1) in {path}")
        sitk_img = sitk.Extract(sitk_img, size[:3] + [0], [0, 0, 0, 0])

    # SimpleITK array is (z, y, x); ANTs expects (x, y, z), so transpose axes.
    arr = np.transpose(sitk.GetArrayFromImage(sitk_img), (2, 1, 0)).astype('float32')
    img = ants.from_numpy(
        arr,
        spacing=tuple(sitk_img.GetSpacing()),
        origin=tuple(sitk_img.GetOrigin()),
        direction=np.array(sitk_img.GetDirection()).reshape(3, 3),
    )
    return ants.reorient_image2(img, target_orientation)

def save_batch_mri(image_references:Union[np.ndarray, ants.ANTsImage],name:str = None,output_path:str = None,file_format:str = '.npz',verbose=0):

    '''

    Save a batch of MRIs in memory to files.

    Parameters
    ----------

    image_references: Dictionary containing the images and their reference names.

    name: name of the main file.

    output_path: directory folder to save the files.

    file_format: file format of the image. Can be saved as a compressed numpy array (.npz) or a compressed Nifti image (.nii.gz)

    '''

    for key,img in image_references.items():
        mri_name = name + '_' + key 
        save_mri(image = img,name = mri_name,output_path=output_path,file_format=file_format,verbose=verbose)
 
def save_mri(image:Union[np.ndarray, ants.ANTsImage],name:str = None,output_path:str = None,file_format:str = '.npz',verbose=1):
    
    '''

    Save image in memory to a file.

    Parameters
    ----------
    
    image: image object to save. Can be either a numpy array or an ANTs image.

    name: name of the file.

    output_path: directory folder to save the file.

    file_format: file format of the image. Can be saved as a compressed numpy array (.npz) or a compressed Nifti image (.nii.gz)

    '''
    if not output_path.endswith('/'): output_path = output_path + '/'
    output_file_path = output_path + name + file_format
    
    if file_format  == '.npz':
        if type(image) is not np.ndarray: image = image.numpy()
        np.savez_compressed(output_file_path ,image)
        # image = ants.from_numpy(image)
    elif file_format == '.nii.gz':
        if type(image) is not ants.ANTsImage: image = ants.from_numpy(image) 
        image.to_file(output_file_path)
    if verbose > 0:
        print("Image saved at:",output_file_path)
    return output_file_path

def load_mri(path:str,as_ants=False):
    '''
    Load image from path as an ANTsImage or numpy compressed array.

    Accepts three inputs:
      - .npz  -> compressed numpy array (the pipeline's intermediate format)
      - .nii / .nii.gz -> single-file volume via ants.image_read
      - a DICOM series -> a directory of .dcm slices (or a path to one .dcm
        inside it); reassembled into one 3D ANTsImage via ants.dicom_read.
    '''

    if path.endswith(".npz"):
        img= np.load(path)['arr_0']
        if as_ants: img = ants.from_numpy(img)
        return img

    # DICOM: ADNI stores one scan as many .dcm slices in an I<id> folder. Read the
    # whole folder as a single, spatially-sorted 3D volume (see load_dicom_series;
    # ants.dicom_read does not sort by ImagePositionPatient and shreds these series).
    if os.path.isdir(path) or path.lower().endswith(".dcm"):
        return load_dicom_series(path)

    return ants.image_read(path)

def set_env_variables():
    print("Setting ANTs and NiftyReg environment variables...\n")

    os.environ['ANTSPATH'] = '/home/lucasthim1/ants/ants_install/bin'
    os.environ['PATH'] =os.environ['PATH'] +  ":" + os.environ['ANTSPATH']
    os.environ['NIFTYREG_INSTALL'] = '/home/lucasthim1/niftyreg/niftyreg_install'
    os.environ['PATH'] = os.environ['PATH'] +  ":" + os.environ['NIFTYREG_INSTALL'] + '/bin'

def check_mri_integrity(image:Union[np.ndarray, ants.ANTsImage]) -> bool:
    
    if type(image) is not np.ndarray: image = image.numpy()
    
    return image.sum().sum().sum() > 0    


