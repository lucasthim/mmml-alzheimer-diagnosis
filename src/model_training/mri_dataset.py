import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset
from torchvision import transforms, utils
from src.utils.base_mri import load_mri

class MRIDataset(Dataset):

     '''
     Builds a dataset loader component for PyTorch with the MRIs based on the filepath.
     '''

     def __init__(self, reference_table,target_column = 'MACRO_GROUP',path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'):
          
          '''
          Initialization of the component

          Parameters
          ----------

          reference_table: Pandas DataFrame containing the reference for the subjects, images and their labels

          '''
          self.target_column = target_column
          self.reference_table = reference_table
          self.path = path

     def __len__(self):
          'Denotes the total number of samples'
          return self.reference_table.shape[0]

     def __getitem__(self, index):
          'Generates one sample of data'
          
          # Select sample
          sample = self.reference_table.iloc[index]

          # Load data and get label
          # X = torch.load('data/' + ID + '.pt')
          X = load_mri(path=self.path + sample['IMAGE_PATH'])
          
          # transforming to tensor and normalizing image between 0 and 1.
          X = X/X.max()
          y = sample[self.target_column]
          return X, y
