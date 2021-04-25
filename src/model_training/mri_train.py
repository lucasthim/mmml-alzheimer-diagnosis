# %%
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
# from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
from torch.optim import Adam, SGD

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models

from mri_dataset import MRIDataset
from mri_train_test_split import train_test_split_by_subject

# IMG_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'
IMG_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/data_alzheimer/'
MODELS_PATH =  '/content/gdrive/MyDrive/Lucas_Thimoteo/data_alzheimer/models/'

# %%

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

class NeuralNetwork(Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.features = Sequential(
            Conv2d(in_channels =1, out_channels =16, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Conv2d(in_channels =16, out_channels =32, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels =32, out_channels =64, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True),
            Conv2d(in_channels =64, out_channels =128, kernel_size=3, stride=1, padding=0),
            ReLU(inplace=True)
        )
        # self.avgpool = AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = Sequential(
            Linear(in_features=128*7*7, out_features=4096, bias=True),
            ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            Linear(in_features=4096, out_features=1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X = X.reshape(dataloader.batch_size, 1, 100, 100)
        # X  = torch.from_numpy(X)
        
        X = X.to(device)
        y = y.to(device)

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # Compute prediction error
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # backpropagation 
        loss.backward()

        # update optimizer
        optimizer.step()
        
        loss = loss.item() 
        if batch % 100 == 0:
            current = batch * len(X)
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        running_loss += loss 
    
    # TODO: Calculate acc, precision and recall
    running_loss = running_loss/size
    return running_loss, None, None, None

def validate_one_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0

    with torch.set_grad_enabled(False):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            loss = loss.item()
            
            if batch % 100 == 0:
                current = batch * len(X)
                print(f"val_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        running_loss += loss    
    # TODO: Calculate acc, precision and recall
    running_loss = running_loss/size
    return running_loss, None, None, None

def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10):
    train_losses = []
    validation_losses = []
    min_valid_loss = np.inf
    early_stopping_marker = 0

    for epoch in range(max_epochs):
        print('---------------------------------------------------------------------')
        print(f'Running Epoch {epoch + 1} of  {max_epochs}')
        t0 = time.time()
        train_loss,train_acc,train_precision,train_recall = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        validation_loss,validation_acc,validation_precision,validation_recall = validate_one_epoch(validation_dataloader, model, loss_fn, optimizer)
        print('Epoch {} took {} seconds'.format(epoch+1, time.time() - t0))
        print(f"Epoch {epoch}. LOSS:: Train_loss {train_loss:.4f}   Val_loss {validation_loss:.4f}")
        print(f"Accuracy:: Train {train_acc:.4f}  Validation {validation_acc:.4f}")
        print('---------------------------------------------------------------------')
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if early_stopping_epochs > 0:
            # TODO: implement early stopping
            # validation for early stopping
            if min_valid_loss >= validation_loss:
               early_stopping_marker += 1
            else:
                min_valid_loss = validation_loss
            
            if early_stopping_marker == early_stopping_epochs:
                print("Exiting training... It hit early stopping criteria of:",early_stopping_epochs,'epochs')
                print("Saving model at:",MODELS_PATH)
                # TODO: Save model with timestamp and small description of experiment.
                # torch.save(model.state_dict(), 'saved_model.pth')
                break
    
    # plotting the training and validation loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.show()

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # calculate test accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# %% ###################################
# Model creation and execution..........
########################################

df_reference = pd.read_csv(IMG_PATH + "REFERENCE.csv")
df_reference = df_reference.query("MACRO_GROUP in ('AD','CN')")
df_reference.loc[df_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1
df_reference.loc[df_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0

# 9 - Build code to run 3 experiments: AD vs CN; AD vs MCI; CN vs MCI
# 10 - Build code to calculate metrics (acc,auc,precision,recall -> get code from colab notebook JCAE) for 3 experiments: AD vs CN; AD vs MCI; CN vs MCI

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Defining Dataset Generators
df_train_reference, df_test_reference = train_test_split_by_subject(df_reference,test_size = 0.3,labels = [1,0],label_column = 'MACRO_GROUP')
df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.3,labels = [1,0],label_column = 'MACRO_GROUP')

df_train_reference = df_train_reference.iloc[:100]
df_validation_reference = df_validation_reference.iloc[:100]

training_set = MRIDataset(reference_table = df_train_reference.iloc[:100],path=IMG_PATH)
training_generator = DataLoader(training_set, **params)

validation_set = MRIDataset(reference_table = df_validation_reference.iloc[:100],path=IMG_PATH)
validation_generator = DataLoader(validation_set, **params)

test_set = MRIDataset(reference_table = df_test_reference.iloc[:100],path=IMG_PATH)
test_generator = DataLoader(test_set, **params)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# defining the models
custom_nn = NeuralNetwork()
print(custom_nn)

vgg = models.vgg11()
print(vgg)
vgg.features[0] = Conv2d(1,64, 3, stride=1,padding=1)
vgg.classifier[-1] = Linear(in_features=4096, out_features=2,bias=True)
print(vgg)

# Transfer model to cuda
model = vgg.to(device)
# model = custom_nn.to(device)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# defining the loss function
criterion = CrossEntropyLoss()
criterion = criterion.to(device)

# %%

train(train_dataloader=training_generator,
    validation_dataloader=validation_generator,
    model=model,
    loss_fn=criterion,
    optimizer=optimizer,
    max_epochs=10,
    early_stopping_epochs=0
    )
# %%

test(dataloader=test_generator,
    model=model,
    loss_fn=criterion)

# %%