# %%
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

from mri_dataset import MRIDataset
from mri_train_test_split import train_test_split_by_subject

# %%

# 7 - Build training procedure with PyTorch (model is imported from another script)
# 8 - Build code with neural network is separate .py
# 9 - Build code to run 3 experiments: AD vs CN; AD vs MCI; CN vs MCI
# 10 - Build code to calculate metrics (acc,auc,precision,recall -> get code from colab notebook JCAE) for 3 experiments: AD vs CN; AD vs MCI; CN vs MCI

img_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'
df_reference = pd.read_csv(img_path + "REFERENCE.csv")

# df_reference['IMAGE_PATH'] = df_reference['IMAGE_PATH'].str.replace("/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/20210329_coronal_50/",'')
# df_reference['IMAGE_PATH'] = df_reference['IMAGE_PATH'].str.replace("/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/processed/20210329_coronal_50/",'')
# df_reference.to_csv("/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/REFERENCE.csv",index=False)

# Parameters
params = {'batch_size': 60,
          'shuffle': True,
          'num_workers': 4}
max_epochs = 100

df_train_reference, df_test_reference = train_test_split_by_subject(df_reference,test_size = 0.3,labels = ['AD','CN'],label_column = 'MACRO_GROUP')
df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.3,labels = ['AD','CN'],label_column = 'MACRO_GROUP')

# Generators
training_set = MRIDataset(reference_table = df_train_reference,path=img_path)
training_generator = DataLoader(training_set, **params)

validation_set = MRIDataset(reference_table = df_validation_reference,path=img_path)
validation_generator = DataLoader(validation_set, **params)
# %%
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


for X, y in training_generator:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


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
# %%
from src.utils.base_mri import load_mri

df_train_reference['IMAGE_PATH'][0]

path_total = img_path + df_train_reference['IMAGE_PATH'][0]
img = load_mri(path='/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/ADNI_023_S_0058_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20080605143248460_S10335_I108504_coronal_52_flip_vertical.npz')

# load_mri
# %%
df_reference.shape
# %%
import os 
import numpy as np

imgs = np.array(os.listdir('/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'))
imgs.shape

