# %%
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset

from src.utils.base_mri import load_mri
# %%

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


# plt.ion()   # interactive mode

# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
        
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
        
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample