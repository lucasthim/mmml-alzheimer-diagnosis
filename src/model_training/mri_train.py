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

from mri_dataset import MRIDataset
from mri_dataset_generation import generate_mri_dataset_reference

import sys
sys.path.append("./../data_preparation")
from train_test_split import train_test_split_by_subject

sys.path.append("./../models")
from neural_network import NeuralNetwork, SuperShallowCNN,create_adapted_vgg11

# %load_ext autoreload
# %autoreload 2

# Defining global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def run_mris_experiments(orientation_and_slices = [('coronal',list(range(45,56)))],
                          num_repeats = 3,
                          model='shallow_cnn',
                          classes=['AD','CN'],
                          mri_config = {
                            'num_samples':0,
                            'num_rotations':0,
                            'sampling_range':3,
                            'mri_reference':'/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20211012_2041.csv',
                            'output_path':'/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/experiments/',
                            },
                          additional_experiment_params = None,
                          save_path = ''):

    results = []
    for orientation,slices in orientation_and_slices:
        mri_config['orientation'] = orientation
        for ii in range(1,num_repeats+1):
            for slice in slices:
                print("\n--------------------------------------------------------------------")
                print("--------------------------------------------------------------------")
                print(f"Running {orientation} - slice:{slice} with no data augmentation.")
                print("--------------------------------------------------------------------")
                print("--------------------------------------------------------------------\n")
                mri_config['slice'] = slice
                df_ref = generate_mri_dataset_reference(mri_reference_path = mri_config['mri_reference'],
                                    output_path = mri_config['output_path'],
                                    orientation = mri_config['orientation'],
                                    orientation_slice = mri_config['slice'],
                                    num_sampled_images = mri_config['num_samples'],
                                    sampling_range = mri_config['sampling_range'],
                                    num_rotations = mri_config['num_rotations'],
                                    save_reference_file = False)
                run_result = run_cnn_experiment(model = model,
                            model_name = 'cnn_'+orientation+str(slice)+str(ii),
                            classes = classes,
                            mri_reference = df_ref,
                            run_test = False,
                            compute_predictions = False,
                            prediction_dataset_path = '',
                            model_path = '',
                            additional_experiment_params = additional_experiment_params)
                run_result['orientation'] = orientation
                run_result['slice'] = slice
                run_result['run'] = ii
                run_result['RUN_ID'] = orientation+str(slice)+str(ii)
                results.append(run_result)

    df_results = pd.concat(results)
    if save_path != '' and save_path is not None:
        df_results.to_csv(save_path,index=False)
    return df_results


def run_experiments_for_ensemble(orientation_and_slices = [('coronal',list(range(45,56)))],
                          model='shallow_cnn',
                          classes=['AD','CN'],
                          mri_config = {
                            'num_samples':0,
                            'num_rotations':0,
                            'sampling_range':3,
                            'mri_reference':'/content/gdrive/MyDrive/Lucas_Thimoteo/data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20211012_2041.csv',
                            'output_path':'/content/gdrive/MyDrive/Lucas_Thimoteo/data/mri/experiments/',
                            },
                          additional_experiment_params = None,
                          save_path = ''):

    predictions = []
    for orientation,slices in orientation_and_slices:
        mri_config['orientation'] = orientation
        for slice in slices:
            print("\n--------------------------------------------------------------------")
            print("--------------------------------------------------------------------")
            print(f"Running {orientation} - slice:{slice} with no data augmentation.")
            print("--------------------------------------------------------------------")
            print("--------------------------------------------------------------------\n")
            mri_config['slice'] = slice
            df_ref = generate_mri_dataset_reference(mri_reference_path = mri_config['mri_reference'],
                                output_path = mri_config['output_path'],
                                orientation = mri_config['orientation'],
                                orientation_slice = mri_config['slice'],
                                num_sampled_images = mri_config['num_samples'],
                                sampling_range = mri_config['sampling_range'],
                                num_rotations = mri_config['num_rotations'],
                                save_reference_file = False)
            prediction,_ = run_cnn_experiment(model = model,
                        model_name = 'cnn_'+orientation+str(slice),
                        classes = classes,
                        mri_reference = df_ref,
                        run_test = False,
                        compute_predictions = True,
                        prediction_dataset_path = '',
                        model_path = '',
                        additional_experiment_params = additional_experiment_params)
            prediction['orientation'] = orientation
            prediction['slice'] = slice
            prediction['RUN_ID'] = orientation+str(slice)
            predictions.append(prediction)

    df_predictions = pd.concat(predictions)
    # TODO: pivot table to make RUN_ID turn into columns for the prediction scores
    if save_path != '' and save_path is not None:
        df_predictions.to_csv(save_path,index=False)
    return df_predictions


def run_cnn_experiment(model = 'vgg11',
                       model_name = 'vgg11_2048_2048',
                       classes = ['AD','CN'],
                       mri_reference = '',
                       run_test = False,
                       compute_predictions = False,
                       prediction_dataset_path = '',
                       model_path = '',
                       additional_experiment_params = None):
    '''
    Run the MRI classification for AD or CN.

    Parameters
    ----------

    model: Neural network to be trained. Can be 'vgg11' or 'shallow'.
    
    model_name: Name to save the trained model.
    
    classes: classes to filter the dataset. options can be ['AD','CN','MCI']

    mri_reference: Path or file of the MRI reference that will be used to filter the validation/test sets and classes. 

    prediction_dataset_path: '/content/gdrive/MyDrive/Lucas_Thimoteo/mri/processed/',
    
    model_path: Path to save the trained model.
    
    additional_experiment_params: dictionary containing some experiments parameters such as lr (learning rate), batch_size and optimizer.

    '''

    if additional_experiment_params is None:
        additional_experiment_params = {'lr':0.0001,
                             'batch_size':16,
                             'optimizer':'adam',
                             'max_epochs':100,
                             'early_stop':10,
                             'prediction_threshold':0.5}
    if type(mri_reference) == str:
        df_mri_reference = pd.read_csv(mri_reference)
    else:
        df_mri_reference = mri_reference
    
    if type(model) == str:
        model = load_model(model)
    model_name = model_name + datetime.now().strftime("%m%d%Y_%H%M")
    
    optimizer,criterion,prepared_data = setup_experiment(model,classes,df_mri_reference,additional_experiment_params)

    train_metrics,validation_metrics = train(train_dataloader=prepared_data['train_dataloader'],
        validation_dataloader=prepared_data['validation_dataloader'],
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        max_epochs=additional_experiment_params['max_epochs'],
        early_stopping_epochs=additional_experiment_params['early_stop'],
        model_name = model_name,
        model_path=model_path)
    
    cols = train_metrics.keys()
    train_cols = ['train_'+x for x in cols]
    df_results = pd.DataFrame([train_metrics])
    df_results.columns = train_cols
    
    validation_cols = ['validation_'+x for x in cols]
    for col,value in zip(validation_cols,validation_metrics.values()):
        df_results[col] = [value]

    if run_test:
        # model.load_state_dict(torch.load(model_path + model_name+'.pth'))
        model.eval()
        test_metrics = test(dataloader=prepared_data['test_dataloader'],
            model=model,
            loss_fn=criterion,
            return_predictions=False,
            compute_metrics=True)
        test_cols = ['test_'+x for x in cols]
        for col,value in zip(test_cols,test_metrics.values()):
            df_results[col] = value
            
    if compute_predictions:
        df_predictions = compute_predictions_for_dataset(prepared_data,model,criterion,threshold = additional_experiment_params['prediction_threshold'])
        if prediction_dataset_path is not None and prediction_dataset_path != '':
            df_predictions.to_csv(prediction_dataset_path + "PREDICTED_MRI_REFERENCE.csv",index=False)
        return df_predictions,df_results
    return df_results

def setup_experiment(model,classes,df_mri_reference,additional_experiment_params):

    print("Setting up experiment parameters...")

    if additional_experiment_params['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=additional_experiment_params['lr'])
    elif additional_experiment_params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=additional_experiment_params['lr'])
    else:
        optimizer = SGD(model.parameters(), lr=additional_experiment_params['lr'])

    dataset_params = {'batch_size': additional_experiment_params['batch_size'],
            'shuffle': False,
            'num_workers': 4,
            'pin_memory':True}
    
    df_train_reference, df_validation_reference, df_test_reference = return_sets(df_mri_reference,classes)

    # Defining Dataset Generators
    training_set = MRIDataset(reference_table = df_train_reference)
    train_dataloader = DataLoader(training_set, **dataset_params)

    validation_set = MRIDataset(reference_table = df_validation_reference)
    validation_dataloader = DataLoader(validation_set, **dataset_params)

    test_set = MRIDataset(reference_table = df_test_reference)
    test_dataloader = DataLoader(test_set, **dataset_params)
    prepared_data = {
        'train_dataloader':train_dataloader,
        'validation_dataloader':validation_dataloader,
        'test_dataloader':test_dataloader,
        'df_train_reference':df_train_reference,
        'df_validation_reference':df_validation_reference,
        'df_test_reference':df_test_reference
    }

    # pos_weight = torch.ones([1]) * (neg_class/pos_class)
    criterion = BCEWithLogitsLoss()
    criterion = criterion.to(device)

    return optimizer,criterion,prepared_data

def return_sets(df_mri_reference,classes):
    if set(classes) == set(['AD','CN']):
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1
    elif set(classes) == set(['MCI','CN']):
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'CN','MACRO_GROUP'] = 0
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'MCI','MACRO_GROUP'] = 1
    elif set(classes) == set(['MCI','AD']):
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'MCI','MACRO_GROUP'] = 0
        df_mri_reference.loc[df_mri_reference['MACRO_GROUP'] == 'AD','MACRO_GROUP'] = 1

    df_mri_reference = df_mri_reference.loc[df_mri_reference['MACRO_GROUP'].isin([0,1]),:]

    filter_query = "DATASET == 'set' and SLICE == MAIN_SLICE"
    if 'ROTATION_ANGLE' in df_mri_reference.columns:
      filter_query = filter_query + " and (ROTATION_ANGLE == 0 or ROTATION_ANGLE == '0')"

    df_validation_reference = df_mri_reference.query(filter_query.replace('set','validation'))
    df_test_reference = df_mri_reference.query(filter_query.replace('set','test'))
    df_train_reference = df_mri_reference.query("DATASET not in ('validation','test')")

    print("Train size:",df_train_reference.shape[0])
    print("Validation size:",df_validation_reference.shape[0])
    print("Test size:",df_test_reference.shape[0])
    return df_train_reference, df_validation_reference, df_test_reference

def compute_predictions_for_dataset(prepared_data, model,criterion,threshold=0.5):

    loaders = [
        prepared_data['train_dataloader'],
        prepared_data['validation_dataloader'],
        prepared_data['test_dataloader']
    ]

    datasets = [
        prepared_data['df_train_reference'],
        prepared_data['df_validation_reference'],
        prepared_data['df_test_reference'],
    ]
    dataset_types = ['train','validation','test']

    print("Saving predictions from trained model...")
    for dataset_type,data_loader,df in zip(dataset_types,loaders,datasets):
        print(f'Computing Predictions for {dataset_type} set.')
        print('dataset size:',df.shape)
        predict_probs,test_metrics = test(dataloader=data_loader,
        model=model,
        loss_fn=criterion,
        compute_metrics=False,
        return_predictions=True)
        predicted_labels = predict_probs >= threshold
        df['CNN_LABEL' ] = predicted_labels
        df['CNN_SCORE' ] = predict_probs

    return pd.concat(datasets)

def load_model(model_type='shallow'):
    print("Loading untrained model...")
    if model_type == 'vgg11':
        vgg = adapt_vgg(models.vgg11())
        model = vgg.to(device)
    
    elif model_type == 'vgg11_bn':
        vgg11_bn = adapt_vgg(models.vgg11_bn())
        model = vgg11_bn.to(device)

    elif model_type == 'vgg13_bn':
        vgg13_bn = adapt_vgg(models.vgg13_bn())
        model = vgg13_bn.to(device)

    elif model_type == 'vgg16_bn':
        vgg16_bn = adapt_vgg(models.vgg16_bn())
        model = vgg16_bn.to(device)

    elif model_type == 'vgg19_bn':
        vgg19_bn = adapt_vgg(models.vgg19_bn())
        model = vgg19_bn.to(device)
    
    elif model_type == 'resnet50':
        resnet50 = adapt_resnet(models.resnet50())
        model = resnet50.to(device)
    
    elif model_type == 'resnet101':
        resnet101 = adapt_resnet(models.resnet101())
        model = resnet101.to(device)
    
    elif model_type == 'shallow_cnn':
        custom_nn = NeuralNetwork()
        model = custom_nn.to(device)
    else:
        custom_nn = SuperShallowCNN()
        model = custom_nn.to(device)
        
    print(model)
    print('')
    count_trainable_parameters(model)
    return model

def adapt_vgg(vgg):
    vgg.features[0] = Conv2d(1,64, 3, stride=1,padding=1)
    vgg.classifier[-1] = Linear(in_features=4096, out_features=1,bias=True)
    return vgg

def adapt_resnet(resnet):
    resnet.conv1 = Conv2d(1,64, 7, stride=2,padding=3)
    resnet.fc = Sequential(
    Linear(in_features=2048, out_features=1000, bias=True),
    ReLU(inplace=True),
    Dropout(p=0.5, inplace=False),
    Linear(in_features=1000, out_features=1, bias=True)
)
    return resnet
def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10,
            model_name = 'experiment',
            model_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/mmml-alzheimer-diagnosis/models/'):

    train_losses = []
    validation_losses = []
    train_aucs = []
    validation_aucs = []
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
        train_aucs.append(train_metrics['auc'])
        validation_aucs.append(validation_metrics['auc'])

        if best_validation_auc >= validation_metrics['auc']:
            early_stopping_marker += 1
        else:
            best_epoch = epoch+1
            best_validation_auc = validation_metrics['auc']
            early_stopping_marker = 0
            best_model_params = model.state_dict()
            best_validation_metrics = validation_metrics
            best_validation_loss = validation_loss
            best_train_metrics = train_metrics
            best_train_loss = train_loss

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

    plot_metric(metric='Loss',train_metric=train_losses,validation_metric= validation_losses)    
    plot_metric(metric='AUC',train_metric=train_aucs,validation_metric= validation_aucs)    
    print('\n-------------------------------')
    print(f"Best metrics for validation set on Epoch {best_epoch}:")
    print_metrics(best_validation_metrics,best_validation_loss)
    print('-------------------------------\n')
    
    return best_train_metrics,best_validation_metrics

def plot_metric(metric,train_metric, validation_metric):
    plt.plot(train_metric, label=f'Train {metric}')
    plt.plot(validation_metric, label=f'Validation {metric}')
    plt.legend()
    plt.title(f"Train vs Validation {metric}")
    plt.show()

def test(dataloader,model,loss_fn,compute_metrics = True, return_predictions = False,dataset_type = 'test'):
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

        test_metrics = None
        if compute_metrics:
            test_loss /= size
            print(f"Performance for {dataset_type} set:")
            test_metrics = compute_metrics_binary(y_true = true_labels, y_pred = predicted_labels,threshold = 0.5,verbose=0)
            print_metrics(test_metrics,test_loss,validation_metrics = None)

        if return_predictions:
            return y_predict_probabilities.cpu().detach().numpy().ravel(),test_metrics
        
        return test_metrics

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
