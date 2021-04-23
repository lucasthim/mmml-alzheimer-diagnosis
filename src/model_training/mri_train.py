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

# img_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'
img_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data_alzheimer/'
df_reference = pd.read_csv(img_path + "REFERENCE.csv")

# Parameters
params = {'batch_size': 64,
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
            nn.Conv2d(in_channels =1, out_channels =16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels =1, out_channels =32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels =1, out_channels =64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels =1, out_channels =128, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer,max_epochs=100,early_stopping = 10):
    # Loop over epochs
    for epoch in range(max_epochs):
        
        # train
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if early_stopping > 0:
            # TODO: implement early stopping
            # validation for early stopping
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in validation_generator:
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)



def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# %%

model = NeuralNetwork().to(device)
print(model)
train()
test()