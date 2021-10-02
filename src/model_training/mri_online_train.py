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
from mri_dataset import MRIDataset

import sys
sys.path.append("./../data_preparation")
from train_test_split import train_test_split_by_subject

sys.path.append("./../models")
from neural_network import NeuralNetwork,create_adapted_vgg11

# %load_ext autoreload
# %autoreload 2

# Defining global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# %%

def run_cnn_experiment(model_type = 'vgg11',
                       model_name = 'vgg11_2048_2048',
                       model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
                       image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/coronal_50_all_4155_images/',
                       ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
                       mri_orientation = 'coronal',
                       mri_slice = 50,
                       classes = [1,0],
                       prediction_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
                       experiment_params = None):
    '''
    Run the MRI classification for AD or CN.

    Parameters
    ----------

    model_type: Neural network to be trained. Can be 'vgg11' or 'shallow'.
    
    model_name: Name to save the trained model.

    model_path: Path to save the trained model.

    image_path: path to load the images for the experiment.

    ensemble_reference_path: Path to the ensemble reference file that will be used to filter the validation and test sets. 
    
    mri_orientation: Orientation which the 2D slice was obtained. Can be 'coronal', 'sagittal' or 'axial',
    
    mri_slice: 2D slice that was extracted from the original 3D MRI,
    
    classes: classes of interest to filter the dataset,
    
    prediction_dataset_path: '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
    
    experiment_params: dictionary containing some experiments parameters such as lr (learning rate), batch_size and optimizer.

    '''

    if experiment_params is None:
        experiment_params = {'lr':0.0001,
                             'batch_size':16,
                             'optimizer':'adam'}
    
    model = load_model(model_type)
    optimizer,criterion,prepared_data = setup_experiment(experiment_params,model,classes,image_path,ensemble_reference_path,mri_slice,mri_orientation)

    model_name = model_name + '_'+ mri_orientation + datetime.now().strftime("%m%d%Y_%H%M")
    train(train_dataloader=prepared_data['train_dataloader'],
        validation_dataloader=prepared_data['validation_dataloader'],
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        max_epochs=100,
        early_stopping_epochs=5,
        model_name = model_name,
        model_path=model_path,
        image_path=image_path)

    model.load_state_dict(torch.load(model_path + model_name+'.pth'))
    model.eval()
    test(dataloader=prepared_data['test_dataloader'],
        model=model,
        loss_fn=criterion)

    compute_predictions_for_dataset(prepared_data,model,criterion,prediction_dataset_path,mri_orientation)

def run_vgg_experiment():
    orientation = 'coronal'
    model_name = '02_vgg11_classifier_2048_2048_'+orientation + '_50_old_augmentation_' + datetime.now().strftime("%m%d%Y_%H%M")
    model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'
    img_path = IMG_PATH
    final_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/'
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
    final_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/'

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
    test(dataloader=test_dataloader,
        model=model,
        loss_fn=criterion)

    compute_predictions_for_dataset(prepared_data,model,criterion,final_dataset_path,orientation)

def setup_experiment(experiment_params,model,classes,image_path,ensemble_reference_path,mri_slice,mri_orientation):

    print("Setting up experiment parameters...")

    if experiment_params['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=experiment_params['lr'])
    elif experiment_params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=experiment_params['lr'])
    else:
        optimizer = SGD(model.parameters(), lr=experiment_params['lr'])

    dataset_params = {'batch_size': experiment_params['batch_size'],
            'shuffle': False,
            'num_workers': 32}
    prepared_data = prepare_dataset_for_training(img_path =image_path,ensemble_reference_path =  ensemble_reference_path,classes = classes,mri_slice = mri_slice,mri_orientation=mri_orientation, dataset_params=dataset_params)
    
    pos_class = classes[0]
    neg_class = classes[1]

    positives = prepared_data['df_train_reference'].query("MACRO_GROUP == @pos_class").shape[0]
    negatives = prepared_data['df_train_reference'].query("MACRO_GROUP == @neg_class").shape[0]
    pos_weight = torch.ones([1]) * (neg_class/pos_class)

    criterion = BCEWithLogitsLoss()
    criterion = criterion.to(device)

    return optimizer,criterion,prepared_data

def prepare_dataset_for_training(img_path,ensemble_reference_path,classes,mri_orientation, mri_slice, dataset_params):
    
    df_reference = pd.read_csv(img_path + "REFERENCE.csv", engine='python')
    df_reference.loc[df_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1
    df_reference.loc[df_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0
    df_reference.loc[df_reference['MACRO_GROUP'] == 'MCI','MACRO_GROUP'] = 2
    df_reference = df_reference.query("MACRO_GROUP in @classes")
    df_reference['IMAGE_NAME']  = [x.split('/')[-1] for x in df_reference['IMAGE_PATH']]
    df_reference['IMAGE_PATH']  = img_path + df_reference['IMAGE_PATH']

    df_ensemble_reference = pd.read_csv(ensemble_reference_path, engine='python')
    df_ensemble_reference['IMAGE_DATA_ID'] = 'I' + df_ensemble_reference['IMAGEUID'].astype(str)
    validation_images = df_ensemble_reference.query("DATASET == 'validation'")['IMAGE_DATA_ID'].unique()
    test_images = df_ensemble_reference.query("DATASET == 'test'")['IMAGE_DATA_ID'].unique()
    
    df_validation_reference = df_reference.query("IMAGE_DATA_ID in @validation_images")
    df_test_reference = df_reference.query("IMAGE_DATA_ID in @test_images")
    df_train_reference = df_reference.query("IMAGE_DATA_ID not in @test_images and IMAGE_DATA_ID not in @validation_images")
    
    # df_reference = df_reference.query("SKIP_IMAGE == False")
    # print(df_reference['SKIP_IMAGE'].sum())

    # df_train_reference, df_test_reference = train_test_split_by_subject(df_reference,test_size = 0.2,labels = classes,label_column = 'MACRO_GROUP')
    # df_train_reference, df_validation_reference = train_test_split_by_subject(df_train_reference,test_size = 0.25,labels = classes,label_column = 'MACRO_GROUP')
    slice_filter = mri_orientation + '_' +str(mri_slice)

    # TODO: Temporary, remove after all new images get uploaded
    df_train_reference = df_validation_reference.query("not IMAGE_NAME.str.contains('rot')",engine='python').sort_values("IMAGE_DATA_ID")
    
    df_validation_reference = df_validation_reference.query("not IMAGE_NAME.str.contains('rot') and IMAGE_NAME.str.contains(@slice_filter)",engine='python').sort_values("IMAGE_DATA_ID")
    df_test_reference = df_test_reference.query("not IMAGE_NAME.str.contains('rot') and IMAGE_NAME.str.contains(@slice_filter)",engine='python').sort_values("IMAGE_DATA_ID")

    print("Train size:",df_train_reference.shape[0])
    print("Validation size:",df_validation_reference.shape[0])
    print("Test size:",df_test_reference.shape[0])

    # Defining Dataset Generators
    training_set = MRIDataset(reference_table = df_train_reference,path=img_path)
    train_dataloader = DataLoader(training_set, **dataset_params)

    validation_set = MRIDataset(reference_table = df_validation_reference,path=img_path)
    validation_dataloader = DataLoader(validation_set, **dataset_params)

    test_set = MRIDataset(reference_table = df_test_reference,path=img_path)
    test_dataloader = DataLoader(test_set, **dataset_params)
    prepared_data = {
        'train_dataloader':train_dataloader,
        'validation_dataloader':validation_dataloader,
        'test_dataloader':test_dataloader,
        'df_train_reference':df_train_reference,
        'df_validation_reference':df_validation_reference,
        'df_test_reference':df_test_reference
    }
    return prepared_data

def compute_predictions_for_dataset(prepared_data, model,criterion,final_dataset_path,orientation,threshold=0.5):

    loaders = list(prepared_data.values())[:3]
    datasets = list(prepared_data.values())[3:]
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
        df['DATASET'] = dataset_type
        print(predict_probs.shape)
    df_reference_with_prediction = pd.concat(datasets)
    df_reference_with_prediction.to_csv(final_dataset_path + 'MRI_REFERENCE_PREDICTIONS_' + orientation + '.csv',index=False)
    print("Reference file with predictions saved at:",final_dataset_path + 'MRI_REFERENCE_PREDICTIONS_' + orientation + '.csv')

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

def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10,
            model_name = 'experiment',
            model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
            image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/coronal_50_all_4155_images/'):
    train_losses = []
    validation_losses = []
    best_epoch = 0
    best_validation_auc = 0
    early_stopping_marker = 0
    best_model_params = model.state_dict()
    best_validation_metrics = None
    best_validation_loss = None
    for epoch in range(max_epochs):
        t0 = time.time()
        
        print('\n---------------------------------------------------------------------')
        print(f'Running Epoch {epoch + 1} of  {max_epochs}')
        
        train_loss,train_metrics = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        validation_loss, validation_metrics = validate_one_epoch(validation_dataloader, model, loss_fn, optimizer)
        
        print_metrics(train_metrics,train_loss,validation_metrics,validation_loss)
        print('\nEpoch {} took'.format(epoch+1),'%3.2f seconds' % (time.time() - t0))
        print('---------------------------------------------------------------------')
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if best_validation_auc >= validation_metrics['auc']:
            early_stopping_marker += 1
        else:
            best_epoch = epoch+1
            best_validation_auc = validation_metrics['auc']
            early_stopping_marker = 0
            best_model_params = model.state_dict()
            best_validation_metrics = validation_metrics
            best_validation_loss = validation_loss
            print('Best validation AUC so far: %1.4f' % best_validation_metrics['auc'])
        
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
    print_metrics(best_validation_metrics,best_validation_loss)
    print('-------------------------------\n')
    return None

def test(dataloader,model,loss_fn,skip_compute_metrics = False, return_predictions = False,dataset_type = 'test'):
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
            print(f"Performance for {dataset_type} set:")
            test_metrics = compute_metrics_binary(y_true = true_labels, y_pred = predicted_labels,threshold = 0.5,verbose=0)
            print_metrics(test_metrics,test_loss,validation_metrics = None)

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
    train_metrics = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
    running_loss = running_loss/size
    return running_loss, train_metrics

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

        validation_metrics = compute_metrics_binary(y_pred = predicted_labels,y_true = true_labels,threshold = 0.5,verbose=0)
        running_loss = running_loss/size
        
        return running_loss, validation_metrics

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
    metrics = {
        'auc':auc,
        'accuracy':accuracy,
        'f1score':f1score,
        'precision':precision,
        'recall':recall,
        'conf_mat':conf_mat
    }
    return metrics

def print_metrics(train_metrics,train_loss,validation_metrics = None,validation_loss = None):
    
    if validation_metrics is not None:

        print(f"Loss::      Train {train_loss:.4f}      Validation {validation_loss:.4f}")
        print(f"AUC::       Train {train_metrics['auc']:.4f}      Validation {validation_metrics['auc']:.4f}")
        print(f"Accuracy::  Train {train_metrics['accuracy']:.4f}      Validation {validation_metrics['accuracy']:.4f}")
        print(f"F1::        Train {train_metrics['f1score']:.4f}      Validation {validation_metrics['f1score']:.4f}")
        print(f"Precision:: Train {train_metrics['precision']:.4f}      Validation {validation_metrics['precision']:.4f}")
        print(f"Recall::    Train {train_metrics['recall']:.4f}      Validation {validation_metrics['recall']:.4f}")
        print("Validation Confusion Matrix:\n", validation_metrics['conf_mat'])
    else:
        print(f"Loss::      {train_loss:.4f}")
        print(f"AUC::       {train_metrics['auc']:.4f}")
        print(f"Accuracy::  {train_metrics['accuracy']:.4f}")
        print(f"F1::        {train_metrics['f1score']:.4f}")
        print(f"Precision:: {train_metrics['precision']:.4f}")
        print(f"Recall::    {train_metrics['recall']:.4f}")
        print("Confusion Matrix:\n", train_metrics['conf_mat'])

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
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("Coronal 50 experiment....")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    run_cnn_experiment(model_type = 'shallow',
                       model_name = 'shallow_cnn',
                       model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
                       image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/coronal_50_67K_images_20210523/',
                       ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
                       mri_orientation = 'coronal',
                       mri_slice = 50,
                       classes = [1,0],
                       prediction_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
                       experiment_params = {'lr':0.0001,
                                            'batch_size':32,
                                            'optimizer':'adam'})
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("Sagittal 25 experiment....")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    
    # run_cnn_experiment(model_type = 'shallow',
    #                     model_name = 'shallow_cnn',
    #                     model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
    #                     image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/sagittal_25_all_4155_images/',
    #                     ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
    #                     mri_orientation = 'sagittal',
    #                     mri_slice = 25,
    #                     classes = [1,0],
    #                     prediction_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
    #                     experiment_params = {'lr':0.0001,
    #                                             'batch_size':16,
    #                                             'optimizer':'adam'})

    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("Axial 25 experiment....")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")

    # run_cnn_experiment(model_type = 'shallow',
    #                     model_name = 'shallow_cnn',
    #                     model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
    #                     image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/axial_25_all_4155_images/',
    #                     ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
    #                     mri_orientation = 'axial',
    #                     mri_slice = 25,
    #                     classes = [1,0],
    #                     prediction_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
    #                     experiment_params = {'lr':0.0001,
    #                                             'batch_size':16,
    #                                             'optimizer':'adam'})
                                                
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("Axial 75 experiment....")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")
    # print("----------------------------------------------------")

    # run_cnn_experiment(model_type = 'shallow',
    #                     model_name = 'shallow_cnn',
    #                     model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/',
    #                     image_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/processed/axial_75_all_4155_images/',
    #                     ensemble_reference_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
    #                     mri_orientation = 'axial',
    #                     mri_slice = 75,
    #                     classes = [1,0],
    #                     prediction_dataset_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/',
    #                     experiment_params = {'lr':0.0001,
    #                                             'batch_size':16,
    #                                             'optimizer':'adam'})
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
