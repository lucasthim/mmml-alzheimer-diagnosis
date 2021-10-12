import warnings
warnings.filterwarnings("ignore")

from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np

class MRIDataset(Dataset):

     '''
     Builds a dataset loader component for PyTorch with the MRIs based on the filepath.
     '''

     def __init__(self, reference_table,target_column = 'MACRO_GROUP'):
          
          '''
          Initialization of the component

          Parameters
          ----------

          reference_table: Pandas DataFrame containing the reference for the subjects, images and their labels

          '''
          self.target_column = target_column
          self.reference_table = reference_table
          # self.transform_train = self.T.Compose([
          #      transforms.RandomCrop(32, padding=4),
          #      transforms.RandomRotation(degrees=15),
          #      transforms.ToTensor(),
          #      transforms.Normalize(rgb_mean, rgb_std),
          # ])
     def __len__(self):
          'Denotes the total number of samples'
          return self.reference_table.shape[0]

     def __getitem__(self, index):
          'Generates one sample of data'
          
          # Select sample
          sample = self.reference_table.iloc[index]

          # Load data and get label
          X = np.load(sample['IMAGE_PATH'])['arr_0']

          if 'ROTATION_ANGLE' in sample.index and sample['ROTATION_ANGLE'] != 0:
               X = ndimage.rotate(X, sample['ROTATION_ANGLE'], reshape=False)
                         
          X = X/X.max()
          y = sample[self.target_column]
          return X, y
