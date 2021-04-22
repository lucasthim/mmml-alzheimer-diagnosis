import torch
from torch.utils import data

from mri_dataset import MRIDataset
from torch.utils.data import Dataset, DataLoader

# TODO: This file will separate training and test and train the model. the training and validation has to be separate as well

# TODO: Add code to build training/validation/test sets

# TODO: Think about to put together the train_test_split code with the images saved with their labels.
# Suggestion:
# Read the dir with the images. Group images by patient. Apply random split by patients.
# Create dict with {'train':[subject IDs],'validation':[subject IDs],'test':[subject IDs]} and save dict as json
# If json exists in the folder, read it, if not, create the json. 
# This should be the first line, and maybe a recursive function.




# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']} # IDs
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1} # Labels

# Generators
training_set = MRIDataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = MRIDataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)


# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        # [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            # [...]