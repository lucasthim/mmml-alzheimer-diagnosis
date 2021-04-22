import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset

from src.utils.base_mri import load_mri

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

   def __len__(self):
        'Denotes the total number of samples'
        return self.reference_table.shape[0]

   def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        sample = self.reference_table.iloc[index]

        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        X = load_mri(path=sample['IMAGE_PATH'])
        y = sample[self.target_column]

        return X, y
