import os

import ants
import nibabel as nib
import numpy as np



def generate_augmented_images(image_3d,orientation,orientation_slice,num_augmented_images,sampling_range,augmentation_type = 'simple'):
    
    '''
    
    Process to generate augmented images.
    
    Arguments:
    - image_3d: 3D MRI object in memory
    - orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial"
    - orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO:fix future bug if sampling_range is outside of the image
    - augmentation_type: Data Augmentation type. Values can be "simple" or "neighborhood_sampling".
    - num_augmented_images: Number of augmented images to sample
    - sampling_range: range to sample new images, with reference to the orientation_slice.
    
     
    '''
    
    if augmentation_type == 'simple':
        image_2d = slice_image(image_3d,orientation,orientation_slice)
        img_dict = generate_augmented_slice(image_2d)
            
        return img_dict

    elif augmentation_type == 'neighborhood_sampling':
        # TODO: Generate logic to execute neighrborhood_sampling process.
        # TODO: Method to iterate throught samples and generate augmented images.
        pass

    
def generate_augmented_slice(image_2d):
    img_rot_90 = np.rot90(image_2d, k=1, axes=(1,0))
    img_rot_180 = np.rot90(image_2d, k=2, axes=(1,0))
    img_rot_270 = np.rot90(image_2d, k=3, axes=(1,0))
    img_flip_horizonal = np.fliplr(image_2d)
    img_flip_vertical = np.flipud(image_2d)

    img_dict = {
        'rot_90':img_rot_90,
        'rot_180':img_rot_180,
        'rot_270':img_rot_270,
        'flip_horizonal':img_flip_horizonal,
        'flip_vertical':img_flip_vertical
    }
    return img_dict

def sample_from_neighborhood(orientation, orientation_slice):
    pass

def augment_samples():
    # for loop cuting image and executing generate_augmented_slice
    # Put everything into the same dict.
    # think about logic to save them all. And also logic to name files and keys of the dict
    pass

def slice_image(image_3d,orientation,orientation_slice):
    '''
    
    Slice a 3D image and create a 2D image.
    
    '''
    
    if orientation == 'coronal':
        return image_3d[orientation_slice,:,:]
    elif orientation == 'sagittal':
        return image_3d[:,orientation_slice,:]
    elif orientation == 'axial':
        return image_3d[:,:,orientation_slice]
        