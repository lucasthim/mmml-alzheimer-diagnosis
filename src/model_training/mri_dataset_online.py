import warnings
warnings.filterwarnings("ignore")

import ants
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms as T

class MRIDatasetOnline(Dataset):

    '''
    Builds a dataset loader component for PyTorch by creating 2D MRIs on the fly, based on a reference file.
    '''

    def __init__(self, reference_table,target_column = 'MACRO_GROUP'):
        
        '''
        Initialization of the component

        Parameters
        ----------

        reference_table: Pandas DataFrame containing the reference for the subjects, images, orientation, slice, rotation and their labels.
        
        target_column: Column containing information about the target/label of the data.


        '''
        self.reference_table = reference_table
        self.target_column = target_column

    def __len__(self):
        'Denotes the total number of samples'
        return self.reference_table.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        sample = self.reference_table.iloc[index]

        # Load data and get label
        # X = np.load(sample['IMAGE_PATH'],allow_pickle=True)['arr_0']
        X = ants.image_read(sample['IMAGE_PATH']).numpy()
        
        X = self._slice_image(image_3d=X,orientation=sample['orientation'],orientation_slice=sample['slice_num'])
        X = self._rotate_image(X,rotation_angle=sample['rotation_angle'])

        if (X.ravel() != X.ravel()).any():
            X[X != X] = np.nanmin(X)
        
        if (X.ravel().sum() == 0):
            print(f"Image {sample['IMAGE_PATH']} is all zeros!")
        
        # transforming to tensor and normalizing image between 0 and 1.
        X = X/X.max()
        y = sample[self.target_column]
        return X, y

        
    def _slice_image(self,image_3d,orientation,orientation_slice):
        
        '''
        
        Slice a 3D image and create a 2D image.
        
        Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
        
        Axis orientation:
        0 - Sagittal
        1 - Coronal
        2 - Axial
        
        Parameters
        ----------
        
        
        image_3d: 3D MRI object in memory as numpy array.
        
        orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial".
        
        orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
        
        Returns
        ----------
        Returns a 2D image.
        
        '''
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

    def _rotate_image(self,image_2d,rotation_angle):
        return ndimage.rotate(image_2d, rotation_angle, reshape=False)


class MRIDatasetOnline2(Dataset):

    '''
    Builds a dataset loader component for PyTorch by creating 2D MRIs on the fly, based on a reference file.
    '''

    def __init__(self, reference_table,target_column = 'MACRO_GROUP'):
        
        '''
        Initialization of the component

        Parameters
        ----------

        reference_table: Pandas DataFrame containing the reference for the subjects, images, orientation, slice, rotation and their labels.
        
        target_column: Column containing information about the target/label of the data.


        '''
        self.reference_table = reference_table
        self.target_column = target_column
        self.transform_train = self.T.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std),
        ])

    def __len__(self):
        'Denotes the total number of samples'
        return self.reference_table.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        sample = self.reference_table.iloc[index]

        # Load data and get label
        X = ants.image_read(sample['IMAGE_PATH']).numpy()
        
        # X = self._slice_image(image_3d=X,orientation=sample['orientation'],orientation_slice=sample['slice_num'])
        # X = self._rotate_image(X,rotation_angle=sample['rotation_angle'])

        # transforming to tensor and normalizing image between 0 and 1.
        X = self.transform_train(X)
        # X = X/X.max()
        y = sample[self.target_column]
        return X, y

        
    def _slice_image(self,image_3d,orientation,orientation_slice):
        
        '''
        
        Slice a 3D image and create a 2D image.
        
        Since ANTsImage to Numpy convertion makes the image lose the reference, we rotate it some times to the correct the axis visualization.
        
        Axis orientation:
        0 - Sagittal
        1 - Coronal
        2 - Axial
        
        Parameters
        ----------
        
        
        image_3d: 3D MRI object in memory as numpy array.
        
        orientation: Orientation to cut the image. Values can be "coronal", "sagittal" or "axial".
        
        orientation_slice: Point to slice the 3D image. Values range from 0 to 100. TODO: fix future bug if sampling_range is outside of the image
        
        Returns
        ----------
        Returns a 2D image.
        
        '''
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

    def _rotate_image(self,image_2d,rotation_angle):
        return ndimage.rotate(image_2d, rotation_angle, reshape=False)
