from __future__ import print_function, division
import os
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

# plt.ion()   # interactive mode


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MRIDataset(data.Dataset):

   '''
   Builds a dataset loader component for PyTorch with the MRIs based on the filepath.
   '''

   def __init__(self, list_IDs, labels):
        
        '''
        Initialization of the component

        Parameters
        ----------

        list:

        '''
        self.labels = labels
        self.list_IDs = list_IDs

   def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

   def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y