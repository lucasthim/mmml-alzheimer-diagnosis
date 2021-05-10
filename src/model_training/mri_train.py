# %%
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

import torch
# from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, BCEWithLogitsLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, AdaptiveAvgPool2d
from torch.optim import Adam, SGD,RMSprop

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.models as models

from mri_dataset import MRIDataset
from mri_train_test_split import train_test_split_by_subject
from src.models.neural_network import NeuralNetwork

%load_ext autoreload
%autoreload 2

# IMG_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210329/'
IMG_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/data_alzheimer/'
MODELS_PATH =  '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'
MODEL_NAME = 'experiment_01_20210429'

# %%

def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10):
    train_losses = []
    validation_losses = []
    best_epoch = 0
    best_validation_auc = 0
    early_stopping_marker = 0
    best_model_params = model.state_dict()
    best_validation_metrics = None
    for epoch in range(max_epochs):
        t0 = time.time()
        
        print('---------------------------------------------------------------------')
        print(f'Running Epoch {epoch + 1} of  {max_epochs}')
        
        train_metrics = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        validation_metrics = validate_one_epoch(validation_dataloader, model, loss_fn, optimizer)
        
        print_metrics(train_metrics,validation_metrics)
        print('\nEpoch {} took'.format(epoch+1),'%3.2f seconds' % (time.time() - t0))
        print('---------------------------------------------------------------------')
        train_loss, validation_loss = train_metrics[0],validation_metrics[0]
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        validation_auc = validation_metrics[1]
        if best_validation_auc >= validation_auc:
            early_stopping_marker += 1
        else:
            best_validation_auc = validation_auc
            early_stopping_marker = 0
            best_model_params = model.state_dict()
            best_validation_metrics = validation_metrics
            best_epoch = epoch+1
            print('Best validation AUC so far: %1.4f' % best_validation_metrics[1])
        
        if early_stopping_epochs > 0:
            if early_stopping_marker == early_stopping_epochs:
                print("\nExiting training... It hit early stopping criteria of:",early_stopping_epochs,'epochs')
                print("Saving model at:",MODELS_PATH)
                # TODO: Save model with timestamp and small description of experiment.
                torch.save(best_model_params, MODELS_PATH + MODEL_NAME + '.pth')
                break

        if (epoch+1) == max_epochs:
            print("Saving model at:",MODELS_PATH,'\n')
            # TODO: Save model with timestamp and small description of experiment.
            torch.save(best_model_params, MODELS_PATH + MODEL_NAME + '.pth')

    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()    
    print('\n-------------------------------')
    print("Best metrics for validation set:")
    print_metrics(best_validation_metrics)
    print('-------------------------------\n')

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(-1,1, 100,100)
            y = y.view(-1,1)
            # y = one_hot(y, num_classes)
            # y = y.view(-1,num_classes)
            y_pred = model(X)
            y = y.type_as(y_pred)

            test_loss += loss_fn(y_pred, y).item()
            true_labels = torch.cat((true_labels,y),0)
            predicted_labels = torch.cat((predicted_labels,y_pred),0)

    print("Performance for Test set:")
    test_loss /= size
    auc, accuracy, f1score, recall, precision,conf_mat = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
    test_metrics = test_loss, auc, accuracy, f1score, recall, precision,conf_mat
    print_metrics(test_metrics,validation_metrics = None)

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1,1, 100,100)
        y = y.view(-1,1)
        # y = one_hot(y, num_classes)
        # y = y.view(-1,num_classes)

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # Compute prediction error
        y_pred = model(X)
        y = y.type_as(y_pred)
        loss = loss_fn(y_pred, y)

        # backpropagation 
        loss.backward()

        # update optimizer
        optimizer.step()
        
        loss = loss.item() 
        # if batch % 2 == 0 and batch > 0:
        #     current = batch * len(X)
        #     print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
        running_loss += loss 
        true_labels = torch.cat((true_labels,y),0)
        predicted_labels = torch.cat((predicted_labels,y_pred),0)
        # print("batch accumulated size:", predicted_labels.size())
    auc, accuracy, f1score, recall, precision, conf_mat = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
    running_loss = running_loss/size
    return running_loss, auc, accuracy, f1score, recall, precision,conf_mat

def validate_one_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)
            X = X.view(-1,1, 100,100)
            y = y.view(-1,1)
            # y = one_hot(y, num_classes)
            # y = y.view(-1,num_classes)

            # Compute prediction error
            y_pred = model(X)
            y = y.type_as(y_pred)
            loss = loss_fn(y_pred, y)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            loss = loss.item()
            
            # if batch % 2 == 0 and batch > 0:
            #     current = batch * len(X)
            #     print(f"val_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss += loss    
            true_labels = torch.cat((true_labels,y),0)
            predicted_labels = torch.cat((predicted_labels,y_pred),0)
            # print("batch accumulated size:", predicted_labels.size())

        auc, accuracy, f1score, recall, precision, conf_mat = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
        running_loss = running_loss/size
        return running_loss, auc, accuracy, f1score, recall, precision,conf_mat

def compute_metrics_binary(y_pred:torch.Tensor,y_true:torch.Tensor,threshold = 0.5,verbose=0):
    y_pred_proba = torch.sigmoid(y_pred)
    y_pred_label = y_pred_proba
    y_pred_label[y_pred_proba >= threshold] = 1
    y_pred_label[y_pred_proba < threshold] = 0
    
    y_true = y_true.cpu().detach().numpy()
    y_pred_label = y_pred_label.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()
    # y_pred_proba = torch.cat((y_pred_proba,1 - y_pred_proba),1).detach().numpy()
    
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred_label)
    f1score = f1_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    conf_mat = confusion_matrix(y_true, y_pred_label)
    if verbose > 0:
        print('----------------')
        print("Total samples in batch:",y_true.shape)
        print("AUC:       %1.3f" % auc)
        print("Accuracy:  %1.3f" % accuracy)
        print("F1:        %1.3f" % f1score)
        print("Precision: %1.3f" % precision)
        print("Recall:    %1.3f" % recall)
        print("Confusion Matrix: \n", conf_mat)
        print('----------------')
    return auc, accuracy, f1score, precision, recall, conf_mat

def print_metrics(train_metrics,validation_metrics = None):
    train_loss, train_auc, train_accuracy, train_f1score, train_precision, train_recall, train_conf_mat = train_metrics
    
    if validation_metrics is not None:
        validation_loss, validation_auc, validation_accuracy, validation_f1score, validation_precision, validation_recall, validation_conf_mat = validation_metrics

        print(f"Loss::      Train {train_loss:.4f}      Validation {validation_loss:.4f}")
        print(f"AUC::       Train {train_auc:.4f}      Validation {validation_auc:.4f}")
        print(f"Accuracy::  Train {train_accuracy:.4f}      Validation {validation_accuracy:.4f}")
        print(f"F1::        Train {train_f1score:.4f}      Validation {validation_f1score:.4f}")
        print(f"Precision:: Train {train_precision:.4f}      Validation {validation_precision:.4f}")
        print(f"Recall::    Train {train_recall:.4f}      Validation {validation_recall:.4f}")
        print("Validation Confusion Matrix:\n", validation_conf_mat)
    else:
        print(f"Loss::      {train_loss:.4f}")
        print(f"AUC::       {train_auc:.4f}")
        print(f"Accuracy::  {train_accuracy:.4f}")
        print(f"F1::        {train_f1score:.4f}")
        print(f"Precision:: {train_precision:.4f}")
        print(f"Recall::    {train_recall:.4f}")
        print("Confusion Matrix:\n", train_conf_mat)

def count_trainable_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print("Total number of trainable parameters:",pp)
    # return pp

# %% 

df_reference = pd.read_csv(IMG_PATH + "REFERENCE.csv")
df_reference = df_reference.query("MACRO_GROUP in ('AD','CN')")
df_reference.loc[df_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1
df_reference.loc[df_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0

df_reference = df_reference #.iloc[:1000]

# Parameters
dataset_params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 32}

# Defining Dataset Generators
df_train_reference, df_test_reference = train_test_split_by_subject(df_reference,test_size = 0.2,labels = [1,0],label_column = 'MACRO_GROUP')
df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.25,labels = [1,0],label_column = 'MACRO_GROUP')

df_train_reference = df_train_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot')",engine='python').sort_values("IMAGE_DATA_ID")
df_validation_reference = df_validation_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot') and UNIQUE_IMAGE_ID.str.contains('_50')",engine='python').sort_values("IMAGE_DATA_ID")
df_test_reference = df_test_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot') and UNIQUE_IMAGE_ID.str.contains('_50')",engine='python').sort_values("IMAGE_DATA_ID")

training_set = MRIDataset(reference_table = df_train_reference,path=IMG_PATH)
training_generator = DataLoader(training_set, **dataset_params)

validation_set = MRIDataset(reference_table = df_validation_reference,path=IMG_PATH)
validation_generator = DataLoader(validation_set, **dataset_params)

test_set = MRIDataset(reference_table = df_test_reference,path=IMG_PATH)
test_generator = DataLoader(test_set, **dataset_params)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# %%

vgg = models.vgg11()
# print(vgg)
vgg.features[0] = Conv2d(1,64, 3, stride=1,padding=1)
vgg.classifier[0] = Linear(in_features=7*7*512, out_features=256,bias=True)
vgg.classifier[3] = Linear(in_features=256, out_features=128,bias=True)
vgg.classifier[-1] = Linear(in_features=128, out_features=1,bias=True)
print(vgg)
# best results with 2048 and 2048 - val_auc = 8121 and test_auc = 0.7787
model = vgg.to(device)

# %%

# custom_nn = NeuralNetwork()
# model = custom_nn.to(device)
# count_trainable_parameters(model)

# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.00001)

# defining the loss function
criterion = BCEWithLogitsLoss()
criterion = criterion.to(device)

# %%

train(train_dataloader=training_generator,
    validation_dataloader=validation_generator,
    model=model,
    loss_fn=criterion,
    optimizer=optimizer,
    max_epochs=100,
    early_stopping_epochs=10
    )

test(dataloader=test_generator,
    model=model,
    loss_fn=criterion)

# %%
from src.models.neural_network import NeuralNetwork
custom_nn = NeuralNetwork()
model = custom_nn.to(device)
print(model)
count_trainable_parameters(model)
MODEL_NAME = 'experiment_01_20210429_baseline_bn_2cnn'

optimizer = RMSprop(model.parameters(), lr=0.0001)
criterion = BCEWithLogitsLoss()
criterion = criterion.to(device)

train(train_dataloader=training_generator,
    validation_dataloader=validation_generator,
    model=model,
    loss_fn=criterion,
    optimizer=optimizer,
    max_epochs=100,
    early_stopping_epochs=10
    )

test(dataloader=test_generator,
    model=model,
    loss_fn=criterion)


# ARCHITECTURE:
# CNN_5x5_16_BN_RELU_MAXPOOL -> CNN_3x3_32_BN_RELU_MAXPOOL -> 2048 > 1024
# data augmentation flip hor,vert and rot90,180 and 270. neigborhood sampling 5
# 0.0001 lr

# Running Epoch 6 of  100
# Loss::      Train 0.0276      Validation 0.0262
# AUC::       Train 0.7447      Validation 0.8848
# Accuracy::  Train 0.7971      Validation 0.9014
# F1::        Train 0.6609      Validation 0.8511
# Precision:: Train 0.7691      Validation 0.8696
# Recall::    Train 0.5795      Validation 0.8333

# -------------------------------
# Best metrics for validation set:
# Loss::      0.0262
# AUC::       0.8848
# Accuracy::  0.9014
# F1::        0.8511
# Precision:: 0.8696
# Recall::    0 .8333
# Confusion Matrix:
#  [[44  3]
#  [ 4 20]]
# -------------------------------

# Performance for Test set:
# Loss::      0.0464
# AUC::       0.8068
# Accuracy::  0.8333
# F1::        0.7500
# Precision:: 0.7826
# Recall::    0.7200
# Confusion Matrix:
#  [[42  5]
#  [ 7 18]]

# SECOND BEST
# CNN_5x5_8 -> CNN_3x3_16 -> 2048 > 1024
# data augmentation flip hor,vert and rot90,180 and 270. neigborhood sampling 5
# 0.001 lr

# -------------------------------
# Best metrics for validation set:
# Loss::      0.0455
# AUC::       0.8839
# Accuracy::  0.8732
# F1::        0.8302
# Precision:: 0.7586
# Recall::    0.9167
# Confusion Matrix:
#  [[40  7]
#  [ 2 22]]
# -------------------------------

# Performance for Test set:
# Loss::      0.1129
# AUC::       0.7562
# Accuracy::  0.7917
# F1::        0.6809
# Precision:: 0.7273
# Recall::    0.6400
# Confusion Matrix:
#  [[41  6]
#  [ 9 16]]

# %%
