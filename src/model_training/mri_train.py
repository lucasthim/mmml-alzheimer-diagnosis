import torch
from torch.utils import data

from mri_dataset import MRIDataset
from mri_train_test_split import train_test_split_by_subject
from torch.utils.data import DataLoader

####################################################################
################# TODO step-by-step ################################
####################################################################

# 7 - Build training procedure with PyTorch (model is imported from another script)
# 8 - Build code with neural network is separate .py
# 9 - Build code to run 3 experiments: AD vs CN; AD vs MCI; CN vs MCI
# 10 - Build code to calculate metrics (acc,auc,precision,recall -> get code from colab notebook JCAE) for 3 experiments: AD vs CN; AD vs MCI; CN vs MCI

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 60,
          'shuffle': True,
          'num_workers': -1}
max_epochs = 100

df_train_reference, df_test_reference = train_test_split_by_subject(df,test_size = 0.3,labels = ['AD','CN'],label_column = ['MACRO_GROUP'])
df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.3,labels = ['AD','CN'],label_column = ['MACRO_GROUP'])

# Generators
training_set = MRIDataset(df_train_reference)
training_generator = DataLoader(training_set, **params)

validation_set = MRIDataset(df_validation_reference)
validation_generator = DataLoader(validation_set, **params)






# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         # TODO: Put neural network here

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             # TODO: Put neural network here



# # Datasets
# partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']} # IDs
# labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1} # Labels

# # Generators
# training_set = MRIDataset(partition['train'], labels)
# training_generator = data.DataLoader(training_set, **params)

# validation_set = MRIDataset(partition['validation'], labels)
# validation_generator = data.DataLoader(validation_set, **params)

# # Loop over epochs
# for epoch in range(max_epochs):
#     # Training
#     for local_batch, local_labels in training_generator:
#         # Transfer to GPU
#         local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#         # Model computations
#         # TODO: Put neural network here

#     # Validation
#     with torch.set_grad_enabled(False):
#         for local_batch, local_labels in validation_generator:
#             # Transfer to GPU
#             local_batch, local_labels = local_batch.to(device), local_labels.to(device)

#             # Model computations
#             # TODO: Put neural network here