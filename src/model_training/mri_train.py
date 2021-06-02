# %%
import time
from datetime import datetime

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
# %%

from src.model_training.mri_dataset import MRIDataset
# from src.model_training.mri_train_test_split import train_test_split_by_subject
from src.data_preparation.train_test_split import train_test_split_by_subject
from src.models.neural_network import NeuralNetwork,create_adapted_vgg11

%load_ext autoreload
%autoreload 2

# Defining global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

img_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210402/'
# MODELS_PATH =  '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'
# MODEL_NAME = 'experiment_01_20210429'
# %%
IMG_PATH = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_67K_images_20210523/'

def run_vgg_experiment():
    orientation = 'coronal'
    model_name = '02_vgg11_classifier_2048_2048_'+orientation + '_50_old_augmentation_' + datetime.now().strftime("%m%d%Y_%H%M")
    model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'
    img_path = IMG_PATH
    final_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/tabular/'
    model = load_model('vgg11')
    optimizer,criterion,prepared_data = setup_experiment(model,img_path,lr=0.00001)
    train_dataloader,validation_dataloader,test_dataloader,_,_,_ = prepared_data

    train(train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        max_epochs=100,
        early_stopping_epochs=5,
        model_name = model_name,
        model_path=model_path,
        image_path=img_path)

    model.load_state_dict(torch.load(model_path + model_name+'.pth'))
    model.eval()
    test(dataloader=test_dataloader,
        model=model,
        loss_fn=criterion)

    compute_predictions_for_dataset(prepared_data,model,criterion,final_dataset_path,orientation)

def run_shallow_cnn_experiment():
    orientation = 'coronal'
    model_name = '02_shallow_cnn_'+ orientation +'_50_old_augmentation_22052021'
    model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'
    img_path = IMG_PATH
    final_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/tabular/'

    model = load_model('shallow')
    optimizer,criterion,prepared_data = setup_experiment(model,img_path,lr=0.0001)
    train_dataloader,validation_dataloader,test_dataloader,_,_,_ = prepared_data
    train(train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        max_epochs=100,
        early_stopping_epochs=5,
        model_name = model_name,
        model_path = model_path,
        image_path = img_path
        )
    model.load_state_dict(torch.load(model_path + model_name +'.pth'))
    model.eval()
    # test(dataloader=test_dataloader,
    #     model=model,
    #     loss_fn=criterion)

    compute_predictions_for_dataset(prepared_data,model,criterion,final_dataset_path,orientation)

def prepare_dataset_for_training(img_path,classes = ['AD','CN'],dataset_params = None,orientation='coronal'):
    
    # TODO: import ENSEMBLE_REFERENCE and filter columns DATASET to get train,val and test sets.
    # WARNING: Validation and test sets should be the same as ensemble. Train set can contain more samples.
    
    df_reference = pd.read_csv(img_path + "REFERENCE.csv")
    df_reference = df_reference.query("MACRO_GROUP in @classes")
    print(df_reference.shape)
    df_reference.loc[df_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1
    df_reference.loc[df_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0
    df_reference.loc[df_reference['MACRO_GROUP'] == 'MCI','MACRO_GROUP'] = 2
    df_reference = df_reference.query("SKIP_IMAGE == False")
    print(df_reference['SKIP_IMAGE'].sum())
    # if orientation == 'coronal':
    #     df_reference1 = df_reference.iloc[:19250].copy()
    #     df_reference2 = df_reference.iloc[22000:].copy()
    #     df_reference = pd.concat([df_reference1,df_reference2])

    # Defining Dataset Generators
    df_train_reference, df_test_reference = train_test_split_by_subject(df_reference,test_size = 0.2,labels = [1,0],label_column = 'MACRO_GROUP')
    df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.25,labels = [1,0],label_column = 'MACRO_GROUP')

    # df_train_reference = df_train_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot')",engine='python').sort_values("IMAGE_DATA_ID")
    df_validation_reference = df_validation_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot') and UNIQUE_IMAGE_ID.str.contains('_50')",engine='python').sort_values("IMAGE_DATA_ID")
    df_test_reference = df_test_reference.query("not UNIQUE_IMAGE_ID.str.contains('flip') and not UNIQUE_IMAGE_ID.str.contains('rot') and UNIQUE_IMAGE_ID.str.contains('_50')",engine='python').sort_values("IMAGE_DATA_ID")

    print("Train size:",df_train_reference.shape[0])
    print("Validation size:",df_validation_reference.shape[0])
    print("Test size:",df_test_reference.shape[0])

    # Parameters
    if dataset_params is None:
        dataset_params = {'batch_size': 16,
                'shuffle': True,
                'num_workers': 32}

    training_set = MRIDataset(reference_table = df_train_reference,path=img_path)
    train_dataloader = DataLoader(training_set, **dataset_params)

    validation_set = MRIDataset(reference_table = df_validation_reference,path=img_path)
    validation_dataloader = DataLoader(validation_set, **dataset_params)

    test_set = MRIDataset(reference_table = df_test_reference,path=img_path)
    test_dataloader = DataLoader(test_set, **dataset_params)
    return train_dataloader,validation_dataloader,test_dataloader,df_train_reference,df_validation_reference,df_test_reference

def load_model(model_type='shallow'):
    print("Loading untrained model...")
    if model_type == 'vgg11':
        vgg = create_adapted_vgg11()
        model = vgg.to(device)
    else:
        custom_nn = NeuralNetwork()
        model = custom_nn.to(device)

    print(model)
    print('')
    count_trainable_parameters(model)
    return model

def setup_experiment(model,img_path,batch_size=16,lr=0.0001):

    print("Setting up experiment parameters...")
    optimizer = RMSprop(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()
    criterion = criterion.to(device)

    dataset_params = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 32}
            
    prepared_data = prepare_dataset_for_training(img_path =img_path ,classes = ['AD','CN'], dataset_params=dataset_params)
    return optimizer,criterion,prepared_data

def compute_predictions_for_dataset(prepared_data, model,criterion,final_dataset_path,orientation,threshold=0.5):

    loaders = [prepared_data[0],prepared_data[1],prepared_data[2]]
    datasets = [prepared_data[3],prepared_data[4],prepared_data[5]]
    dataset_types = ['train','validation','test']
    print("Saving predictions from trained model...")
    for dataset_type,data_loader,df in zip(dataset_types,loaders,datasets):
        print('dataset type:',dataset_type)
        print('dataset size:',df.shape)
        predict_probs = test(dataloader=data_loader,
        model=model,
        loss_fn=criterion,
        skip_compute_metrics=False,
        return_predictions=True)
        predicted_labels = predict_probs >= threshold
        df['DL_PREDICTION_' + orientation] = predicted_labels
        df['DL_PREDICT_PROBA_' + orientation] = predict_probs
        df['DATASET_TYPE'] = dataset_type
        print(predict_probs.shape)
    df_reference_with_prediction = pd.concat(datasets)
    df_reference_with_prediction.to_csv(final_dataset_path + 'MRI_REFERENCE_PREDICTIONS_' + orientation + '.csv',index=False)
    print("Reference file with predictions saved at:",final_dataset_path + 'MRI_REFERENCE_PREDICTIONS_' + orientation + '.csv')

def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10,
            model_name = 'experiment',
            model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
            image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_50_25K_images_20210402/'):
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
            best_epoch = epoch+1
            best_validation_auc = validation_auc
            early_stopping_marker = 0
            best_model_params = model.state_dict()
            best_validation_metrics = validation_metrics
            print('Best validation AUC so far: %1.4f' % best_validation_metrics[1])
        
        if early_stopping_epochs > 0:
            if early_stopping_marker == early_stopping_epochs:
                print("\nExiting training... It hit early stopping criteria of:",early_stopping_epochs,'epochs')
                print("Saving model at:",model_path)
                torch.save(best_model_params, model_path + model_name + '.pth')
                break

        if (best_epoch) == max_epochs:
            print("Saving model at:",model_path,'\n')
            torch.save(best_model_params, model_path + model_name + '.pth')

    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()    
    print('\n-------------------------------')
    print(f"Best metrics for validation set on Epoch {best_epoch}:")
    print_metrics(best_validation_metrics)
    print('-------------------------------\n')
    return None

def test(dataloader,model,loss_fn,skip_compute_metrics = False, return_predictions = False):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)
    y_predict_probabilities = torch.Tensor().to(device)
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
            if return_predictions:
                y_predict_proba = torch.sigmoid(y_pred)
                y_predict_probabilities = torch.cat((y_predict_probabilities,y_predict_proba),0)

        if not skip_compute_metrics:
            test_loss /= size
            print("Performance for Test set:")
            auc, accuracy, f1score, recall, precision,conf_mat = compute_metrics_binary(y_true = true_labels, y_pred = predicted_labels,threshold = 0.5,verbose=0)
            test_metrics = test_loss, auc, accuracy, f1score, recall, precision,conf_mat
            print_metrics(test_metrics,validation_metrics = None)

        if return_predictions:
            return y_predict_probabilities.cpu().detach().numpy().ravel()

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
        running_loss += loss 
        true_labels = torch.cat((true_labels,y),0)
        predicted_labels = torch.cat((predicted_labels,y_pred),0)
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

            y_pred = model(X)
            y = y.type_as(y_pred)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss = loss.item()
            
            running_loss += loss    
            true_labels = torch.cat((true_labels,y),0)
            predicted_labels = torch.cat((predicted_labels,y_pred),0)

        auc, accuracy, f1score, recall, precision, conf_mat = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
        running_loss = running_loss/size
        
        return running_loss, auc, accuracy, f1score, recall, precision,conf_mat

def compute_metrics_binary(y_true:torch.Tensor, y_pred:torch.Tensor, y_pred_proba:torch.Tensor = None,threshold = 0.5,verbose=0):
    
    if y_pred_proba is None:
        y_pred_proba = torch.sigmoid(y_pred)
    y_pred_label = y_pred_proba
    y_pred_label[y_pred_proba >= threshold] = 1
    y_pred_label[y_pred_proba < threshold] = 0
    
    y_true = y_true.cpu().detach().numpy()
    y_pred_label = y_pred_label.cpu().detach().numpy()
    y_pred_proba = y_pred_proba.cpu().detach().numpy()

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

if __name__ == "__main__":
    run_shallow_cnn_experiment()
    # run_vgg_experiment()

# Best result so far...
# dataset type: train
# dataset size: (18756, 18)
# Performance for Test set:
# Loss::      0.0303
# AUC::       0.8921
# Accuracy::  0.9077
# F1::        0.8414
# Precision:: 0.8275
# Recall::    0.8557
# Confusion Matrix:
#  [[12435   957]
#  [  774  4590]]
# (18756, 1)
# dataset type: validation
# dataset size: (187, 18)
# Performance for Test set:
# Loss::      0.0184
# AUC::       0.8918
# Accuracy::  0.8824
# F1::        0.8406
# Precision:: 0.7733
# Recall::    0.9206
# Confusion Matrix:
#  [[107  17]
#  [  5  58]]
# (187, 1)
# dataset type: test
# dataset size: (185, 18)
# Performance for Test set:
# Loss::      0.0301
# AUC::       0.9117
# Accuracy::  0.9135
# F1::        0.8596
# Precision:: 0.8167
# Recall::    0.9074
# Confusion Matrix:
#  [[120  11]
#  [  5  49]]


# Saving predictions from trained model...
# dataset type: train
# dataset size: (27360, 15)
# Performance for Test set:
# Loss::      0.0048
# AUC::       0.9477
# Accuracy::  0.9674
# F1::        0.9442
# Precision:: 0.9974
# Recall::    0.8965
# Confusion Matrix:
#  [[18916    20]
#  [  872  7552]]
# (27360,)
# dataset type: validation
# dataset size: (247, 15)
# Performance for Test set:
# Loss::      0.0587
# AUC::       0.8337
# Accuracy::  0.8543
# F1::        0.7805
# Precision:: 0.7901
# Recall::    0.7711
# Confusion Matrix:
#  [[147  17]
#  [ 19  64]]
# (247,)
# dataset type: test
# dataset size: (271, 15)
# Performance for Test set:
# Loss::      0.0680
# AUC::       0.7402
# Accuracy::  0.8266
# F1::        0.6357
# Precision:: 0.7593
# Recall::    0.5467
# Confusion Matrix:
#  [[183  13]
#  [ 34  41]]
# (271,)

# %%

df_reference = pd.read_csv("/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_67K_REFERENCE.csv")
df_reference2 = pd.read_csv("/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/coronal_25K_REFERENCE.csv")

df_reference['SKIP_IMAGE'] = False
# df_reference.loc[19250:22000,'SKIP_IMAGE'] = True

df_reference_final = df_reference.drop(['MODALITY','FORMAT','DOWNLOADED','SUBJECT_ID'],axis=1)
df_reference2['SKIP_IMAGE'] = False
df_reference_final = pd.concat([df_reference_final,df_reference2])

imgs_to_skip = df_reference_final.query("IMAGE_PATH.str.contains('I85699')",engine='python')['IMAGE_PATH']
df_reference_final.loc[df_reference_final['IMAGE_PATH'].isin(imgs_to_skip),'SKIP_IMAGE'] = True

imgs_to_skip = df_reference_final.query("IMAGE_PATH.str.contains('I382256')",engine='python')['IMAGE_PATH']
df_reference_final.loc[df_reference_final['IMAGE_PATH'].isin(imgs_to_skip),'SKIP_IMAGE'] = True

df_reference_final.to_csv("/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/data/mri/processed/REFERENCE.csv")
# %%
