import os
import random

import ants
import nibabel as nib
import numpy as np



def generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'simple')-> dict:
    
    '''
    
    Process to generate augmented images.
    
    Arguments:
    - image_3d: 3D MRI object in memory
    - orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    - orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
    - num_augmented_images: Number of augmented images to sample
    - sampling_range: range to sample new images, with reference to the orientation_slice.
    - augmentation_type: Data Augmentation type. Values can be "simple" or "neighborhood_sampling".
     
    '''
    image_2d = slice_image(image_3d,orientation,orientation_slice)
    img_dict = generate_augmented_slice(image_2d,orientation,orientation_slice)
    
    if augmentation_type == 'neighborhood_sampling':
        
        samples = sample_from_neighborhood(orientation_slice,sampling_range,num_augmented_images)
        for sample_slice in samples:
            sample_2d = slice_image(image_3d,orientation,sample_slice)
            img_dict.update(generate_augmented_slice(sample_2d,orientation,sample_slice))
    return img_dict
    
def generate_augmented_slice(image_2d,orientation,orientation_slice):
    img_rot_90 = np.rot90(image_2d, k=1, axes=(1,0))
    img_rot_180 = np.rot90(image_2d, k=2, axes=(1,0))
    img_rot_270 = np.rot90(image_2d, k=3, axes=(1,0))
    img_flip_horizonal = np.fliplr(image_2d)
    img_flip_vertical = np.flipud(image_2d)

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
    random.seed(a=None, version=2)
    neighbor_samples = list(set(range(orientation_slice-sampling_range,orientation_slice+sampling_range+1)) - set([orientation_slice]))
    samples = random.sample(neighbor_samples,k=num_augmented_images)
    return samples

def slice_image(image_3d,orientation,orientation_slice):
    '''
    
    Slice a 3D image and create a 2D image.
    
    '''
    
    if orientation == 'coronal' or orientation == 0:
        return image_3d[orientation_slice,:,:]
    elif orientation == 'sagittal' or orientation == 1:
        return image_3d[:,orientation_slice,:]
    elif orientation == 'axial' or orientation == 2:
        return image_3d[:,:,orientation_slice]
        