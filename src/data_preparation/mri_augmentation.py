import os
import random

import ants
import nibabel as nib
import numpy as np



def generate_augmented_images(image_3d:ants.ANTsImage,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'simple')-> dict:
    
    '''
    
    Process to generate augmented MRI images. 

    Main steps:

    1 - Orientation selection.

    2 - Slice Selection.

    3 - Sampling new images around the chosen slice, within a given range.

    4 - Apply flipping and rotation to the chosen images.


    Parameters
    ----------
    
    image_3d: 3D MRI ANTsImage object in memory
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
    
    num_augmented_images: Number of augmented images to sample
    
    sampling_range: range to sample new images, with reference to the orientation_slice.
    
    augmentation_type: Data Augmentation type. Values can be "simple" or "neighborhood_sampling".


    Returns
    ----------

    A dictionary containing all the augmented images. The keys are the image names plus augmentation type and the values are the image objects.
    '''

    image_2d = slice_image(image_3d,orientation,orientation_slice)
    img_dict = generate_augmented_slice(image_2d,orientation,orientation_slice)
    
    if augmentation_type == 'neighborhood_sampling':
        
        samples = sample_from_neighborhood(orientation_slice,sampling_range,num_augmented_images)
        for sample_slice in samples:
            sample_2d = slice_image(image_3d,orientation,sample_slice)
            augmented_slice = generate_augmented_slice(sample_2d,orientation,sample_slice)
            img_dict.update(augmented_slice)
    return img_dict
    
def generate_augmented_slice(image_2d:np.ndarray,orientation,orientation_slice)-> dict:
    
    '''
    Data augmentation with 90, 180, 270 rotations and also horizontal and vertical flipping.
    
    Parameters
    ----------
    
    image_2d: 2D image object
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
    
    num_augmented_images: Number of augmented images to sample

    Returns
    ----------

    A dictionary containing all the augmented images. The keys are the image names plus augmentation type and the values are the image objects.
    '''
    # TODO:change these 5 augmentations for random rotation between -15 and +15 degrees.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    
    img_rot_90 = np.rot90(image_2d, k=1, axes=(1,0)).copy()
    img_rot_180 = np.rot90(image_2d, k=2, axes=(1,0)).copy()
    img_rot_270 = np.rot90(image_2d, k=3, axes=(1,0)).copy()
    img_flip_horizonal = np.fliplr(image_2d).copy()
    img_flip_vertical = np.flipud(image_2d).copy()

    img_key = orientation + '_' + str(orientation_slice)
    img_dict = {
        img_key: image_2d,
        img_key + '_rot_90':img_rot_90,
        img_key + '_rot_180':img_rot_180,
        img_key + '_rot_270':img_rot_270,
        img_key + '_flip_horizonal':img_flip_horizonal,
        img_key + '_flip_vertical':img_flip_vertical
    }
    return img_dict

def sample_from_neighborhood(orientation_slice,sampling_range,num_augmented_images):
    
    '''
    Sample new slices around a reference point of a 3D image. Returns an array with the index of the new sampled slices.
    
    Parameters
    ----------
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
    
    num_augmented_images: Number of augmented images to sample
    
    sampling_range: range to sample new images, with reference to the orientation_slice.
    

    Returns
    ----------

    List of sampled images.
    '''
    
    random.seed(a=None, version=2)
    neighbor_samples = list(set(range(orientation_slice-sampling_range,orientation_slice+sampling_range+1)) - set([orientation_slice]))
    samples = random.sample(neighbor_samples,k=num_augmented_images)
    return samples

def slice_image(image_3d: ants.ANTsImage,orientation,orientation_slice):
    
    '''
    
    Slice a 3D image and create a 2D image.
    
    Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
    
    Axis orientation:
    0 - Sagittal
    1 - Coronal
    2 - Axial
    
    Parameters
    ----------
    
    
    image_3d: 3D MRI object in memory
    
    orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    
    orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image

    Returns
    ----------
    Returns a 2D image.
    
    '''
    image_3d = image_3d.numpy()
    if orientation == 'sagittal':
        rot = np.rot90(image_3d, k=3, axes=(1,2)).copy()
        rot = np.rot90(rot, k=2, axes=(0,2)).copy()
        return rot[orientation_slice,:,:]
    
    elif orientation == 'coronal':
        rot = np.rot90(image_3d, k=3, axes=(0,2)).copy()
        return rot[:,orientation_slice,:]
    
    elif orientation == 'axial':
        rot = np.rot90(image_3d, k=3, axes=(0,1)).copy()        
        return rot[:,:,orientation_slice]
