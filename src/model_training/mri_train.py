import sys
import time
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.nn import Linear, ReLU, BCEWithLogitsLoss, Sequential, Conv2d, Dropout
from torch.optim import Adam, SGD,RMSprop
import torchvision.models as models

from mri_dataset import MRIDataset
from mri_dataset_generation import generate_mri_dataset_reference

sys.path.append("./../models")
from neural_network import load_model,load_trained_model,device
from loss import WeightedFocalLoss

sys.path.append("./../model_evaluation")
from base_evaluation import compute_metrics_binary


print("Using {} device".format(device))

# %load_ext autoreload
# %autoreload 2


def compute_predictions_for_ensemble(orientations=['coronal','sagittal','axial'],
                                     slices=[70,50,8],
                                     model_types=['vgg19_bn','vgg19_bn','vgg19_bn'],
                                     model_paths=[],
                                     classes=['AD','CN'],
                                     mri_config={},
                                     save_path=''):
    '''
    Compute predictions of orientations and slices for the ensemble experiment.
    '''
    
    dfs=[]
    for model,model_path,orientation,slice in zip(model_types,model_paths,orientations,slices):
        print('\n--------------------------------------------------------------------------------')
        print(f"Evaluating performance on trained {model} for orientation {orientation} and slice {slice}")
        print('--------------------------------------------------------------------------------')
        mri_config['orientation'] = orientation
        mri_config['slice'] = slice
        df = evaluate_trained_model(model=model,
                            model_path=model_path,
                            classes=classes,
                            mri_reference=mri_config)
        
        df['model'] = model
        df['model_path'] = model_path

        dfs.append(df)
    df_predictions = pd.concat(dfs)
    if save_path != '':
        df_predictions.to_csv(save_path,index=False)
    return df_predictions

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
                
                try:
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
                  run_result,_ = run_cnn_experiment(model = model,
                              model_name = 'cnn_'+orientation+str(slice)+str(ii),
                              classes = classes,
                              mri_reference = df_ref,
                              additional_experiment_params = additional_experiment_params)
                  run_result['orientation'] = orientation
                  run_result['slice'] = slice
                  run_result['run'] = ii
                  run_result['RUN_ID'] = orientation+str(slice)+str(ii)
                  results.append(run_result)
                  if save_path != '' and save_path is not None:
                      df_results = pd.concat(results)
                      df_results.to_csv(save_path,index=False)
                except BaseException as err:
                  print(f"Experiment with orientation {orientation} and slice {slice} failed! Moving to the next.")
                  print(f"\n Unexpected {err}, {type(err)}")
                  continue

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
                          save_path = '',
                          model_path=''):

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
            df_results,saved_model_path = run_cnn_experiment(model = model,
                        model_name = 'cnn_'+orientation+'_'+str(slice),
                        classes = classes,
                        mri_reference = df_ref,
                        model_path = model_path+'_'+orientation+'_'+str(slice),
                        additional_experiment_params = additional_experiment_params)
            
            prediction = evaluate_trained_model(model=model,
                           model_path=saved_model_path,
                           classes=classes,
                           mri_reference=df_ref)
            predictions.append(prediction)

    df_predictions = pd.concat(predictions)

    if save_path != '' and save_path is not None:
        df_predictions.to_csv(save_path,index=False)
    return df_predictions

def run_cnn_experiment(model = 'vgg11',
                       model_name = 'vgg11_2048_2048',
                       classes = ['AD','CN'],
                       mri_reference = '',
                      #  compute_predictions = False,
                      #  prediction_dataset_path = '',
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

    compute_predictions: Flag to compute final predictions after model is trained.
    
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
                             'early_stop_metric':'auc',
                             'prediction_threshold':0.5}
    if type(mri_reference) == str:
        df_mri_reference = pd.read_csv(mri_reference)
    else:
        df_mri_reference = mri_reference
    
    # original_model_type = deepcopy(model)
    model = load_model(model)
        
    model_name = model_name + datetime.now().strftime("%m%d%Y_%H%M")
    
    optimizer,criterion,prepared_data = setup_experiment(model,classes,df_mri_reference,additional_experiment_params)

    train_metrics,validation_metrics,saved_model_path = train(train_dataloader=prepared_data['train_dataloader'],
        validation_dataloader=prepared_data['validation_dataloader'],
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        max_epochs=additional_experiment_params['max_epochs'],
        early_stopping_epochs=additional_experiment_params['early_stop'],
        early_stopping_metric = additional_experiment_params['early_stop_metric'],
        model_name = model_name,
        model_path=model_path)
    
    cols = train_metrics.keys()
    train_cols = ['train_'+x for x in cols]
    df_results = pd.DataFrame([train_metrics])
    df_results.columns = train_cols
    
    validation_cols = ['validation_'+x for x in cols]
    for col,value in zip(validation_cols,validation_metrics.values()):
        df_results[col] = [value]
    # if compute_predictions:
    #     model = load_model(original_model_type)
    #     return model,df_results
    #     model = model.load_state_dict(best_model_params,strict=True)
    #     model.to(device)
    #     model.eval()

    #     df_predictions = evaluate_trained_model(model=model,
    #                        classes=classes,
    #                        mri_reference=df_mri_reference)

    #     if prediction_dataset_path != '':
    #         df_predictions.to_csv(prediction_dataset_path + "PREDICTED_MRI_REFERENCE.csv",index=False)
    #     return df_predictions,df_results

    return df_results,saved_model_path

def setup_experiment(model,classes,df_mri_reference,additional_experiment_params):

    print("Setting up experiment parameters...")

    if additional_experiment_params['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=additional_experiment_params['lr'])
    elif additional_experiment_params['optimizer'] == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=additional_experiment_params['lr'])
    else:
        optimizer = SGD(model.parameters(), 
                        lr=additional_experiment_params['lr'],
                        momentum=additional_experiment_params['momentum'])

    dataset_params = {'batch_size': additional_experiment_params['batch_size'],
            'shuffle': True,
            'num_workers': 4,
            'pin_memory':False}
    
    df_train_reference, df_validation_reference, df_test_reference = return_sets(df_mri_reference,classes)

    # Defining Dataset Generators
    training_set = MRIDataset(reference_table = df_train_reference)
    train_dataloader = DataLoader(training_set, **dataset_params)

    validation_set = MRIDataset(reference_table = df_validation_reference)
    validation_dataloader = DataLoader(validation_set, shuffle=False,batch_size=1024)

    test_set = MRIDataset(reference_table = df_test_reference)
    test_dataloader = DataLoader(test_set, shuffle=False,batch_size=1024)
    prepared_data = {
        'train_dataloader':train_dataloader,
        'validation_dataloader':validation_dataloader,
        'test_dataloader':test_dataloader,
        'df_train_reference':df_train_reference,
        'df_validation_reference':df_validation_reference,
        'df_test_reference':df_test_reference
    }

    df_train = df_mri_reference.query("DATASET not in ('validation','test')")

    neg_class = df_train.query("MACRO_GROUP == 0").shape[0]
    pos_class = df_train.query("MACRO_GROUP == 1").shape[0]

    if additional_experiment_params['loss'] == 'FocalLoss':
        alpha =  (pos_class/neg_class)
        criterion = WeightedFocalLoss(alpha=alpha,gamma=additional_experiment_params['loss_gamma'])
    else:
        pos_weight = torch.ones([1]) * (neg_class/pos_class)
        criterion = BCEWithLogitsLoss(pos_weight=pos_weight,reduction='mean')
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

def train(train_dataloader,
            validation_dataloader, 
            model, 
            loss_fn, 
            optimizer,
            max_epochs=100,
            early_stopping_epochs = 10,
            early_stopping_metric = 'auc',
            model_name = 'cnn',
            model_path = ''):

    train_losses = []
    validation_losses = []
    train_aucs = []
    validation_aucs = []
    train_f1s = []
    validation_f1s = []
        
    best_epoch = 0
    best_validation_metric = 0
    early_stopping_marker = 0
    best_model_params = deepcopy(model.state_dict())
    best_validation_metrics = None
    best_validation_loss = None
    model.to(device)
    model.train()
    final_model_path = ''
    for epoch in range(max_epochs):
        t0 = time.time()
        
        print('\n---------------------------------------------------------------------')
        print(f'Running Epoch {epoch + 1} of  {max_epochs}')
        
        train_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer)
        train_metrics,_,_ = evaluate_one_epoch(train_dataloader, model, loss_fn)
        validation_metrics,validation_loss,_ = evaluate_one_epoch(validation_dataloader, model, loss_fn)
        
        print_metrics(train_metrics,train_loss,validation_metrics,validation_loss)
        print('\nEpoch {} took'.format(epoch+1),'%3.2f seconds' % (time.time() - t0))
        print('---------------------------------------------------------------------')
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_aucs.append(train_metrics['auc'])
        validation_aucs.append(validation_metrics['auc'])
        train_f1s.append(train_metrics['f1score'])
        validation_f1s.append(validation_metrics['f1score'])

        if best_validation_metric >= validation_metrics[early_stopping_metric]:
            early_stopping_marker += 1
        else:
            best_epoch = epoch+1
            best_validation_metric = validation_metrics[early_stopping_metric]
            early_stopping_marker = 0
            best_model_params = deepcopy(model.state_dict())
            best_validation_metrics = validation_metrics
            best_validation_loss = validation_loss
            best_train_metrics = train_metrics
            best_train_loss = train_loss

            print('Best validation '+ early_stopping_metric + ' so far: %1.4f' % best_validation_metrics[early_stopping_metric])
        
        if early_stopping_epochs > 0:
            if early_stopping_marker == early_stopping_epochs:
                print("\nExiting training... It hit early stopping criteria of:",early_stopping_epochs,'epochs')
                
                if model_path != '':
                    final_model_path = model_path + model_name + '.pth'
                    print("Saving model at:",final_model_path)
                    torch.save(best_model_params, final_model_path)
                break

        if (best_epoch) == max_epochs:
            if model_path != '':
                final_model_path = model_path + model_name + '.pth'
                print("Saving model at:",final_model_path)
                torch.save(best_model_params, final_model_path)

    print('\n-------------------------------')
    plot_metric(metric='Loss',train_metric=train_losses,validation_metric= validation_losses)    
    print('')
    plot_metric(metric='AUC',train_metric=train_aucs,validation_metric= validation_aucs)    
    print('')
    plot_metric(metric='F1 Score',train_metric=train_f1s,validation_metric= validation_f1s)    
    print('\n-------------------------------')
    
    print(f"Best metrics for validation set on Epoch {best_epoch}:")
    print_metrics(best_validation_metrics,best_validation_loss)
    print('-------------------------------\n')
    print(f"Metrics for train set on Epoch {best_epoch}:")
    print_metrics(best_train_metrics,best_train_loss)
    print('-------------------------------\n')
    
    return best_train_metrics,best_validation_metrics,final_model_path

def plot_metric(metric,train_metric, validation_metric):
    plt.plot(train_metric, label=f'Train {metric}')
    plt.plot(validation_metric, label=f'Validation {metric}')
    plt.legend()
    plt.title(f"Train vs Validation {metric}")
    plt.show()

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    true_labels = torch.Tensor().to(device)
    predicted_labels = torch.Tensor().to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.view(-1,1, 100,100)
        y = y.view(-1,1)

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
    
    running_loss = running_loss/size
    return running_loss

def evaluate_one_epoch(dataloader, model, loss_fn,predictions=False):
    
    '''
    Evaluate a model after an epoch of training. 
    
    Returns metrics, loss and predictions

    '''

    size = len(dataloader.dataset)
    running_loss = 0.0
    true_labels = torch.Tensor().to(device)
    predicted_logits = torch.Tensor().to(device)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)
            X = X.view(-1,1, 100,100)
            y = y.view(-1,1)

            y_pred = model(X)
            y = y.type_as(y_pred)
            loss = loss_fn(y_pred, y).item()
            
            running_loss += loss    
            true_labels = torch.cat((true_labels,y),0)
            predicted_logits = torch.cat((predicted_logits,y_pred),0)
        predicted_probas = torch.sigmoid(predicted_logits)
        metrics = compute_metrics_binary(y_true = true_labels, y_pred_proba = predicted_probas, threshold = 0.5,verbose=0)
        running_loss = running_loss/size
                
        if predictions:
            predicted_probas = predicted_probas.cpu().detach().numpy().ravel()

        return metrics,running_loss,predicted_probas

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

def evaluate_trained_model(model='shallow_cnn',
                           model_path='',
                           classes=['AD','CN'],
                           mri_reference={},
                           save_predictions_path=''):
    '''
    Evaluate trained model on MRI reference dataset. 

    Parameters
    ------------

    model: Model to evaluate. Can be a string indicating model type or a torch object

    model_path: Path to load weights of a trained model. Only works if model is a string.

    classes: Classes to evaluate.

    mri_reference: Reference dataset to evaluate trained model. Can be a dict with information to load the dataset or the dataset itself.

    save_predictions_path: Path to save the predictions generated during the model evaluation.

    Returns
    ----------

    Tuple containing dataset along with model predictions on column CNN_SCORE.

    '''

    if isinstance(mri_reference,pd.DataFrame):
        df_ref = mri_reference.copy()
    else:
        df_ref = generate_mri_dataset_reference(
        mri_reference_path = mri_reference['mri_reference'],
        output_path = mri_reference['output_path'],
        orientation = mri_reference['orientation'],
        orientation_slice = mri_reference['slice'],
        num_sampled_images = mri_reference['num_samples'],
        sampling_range = mri_reference['sampling_range'],
        num_rotations = mri_reference['num_rotations'],
        save_reference_file = False)

    if isinstance(model,str):
        model = load_trained_model(model=model,model_path=model_path);
    df_train_reference, df_validation_reference, df_test_reference = return_sets(df_ref,classes)

    predictions_df=[]
    for set_type,df in zip(['Training','Validation','Test'],[df_train_reference, df_validation_reference, df_test_reference]):

        print(f"\n{set_type} set:")
        _,predictions = evaluate_model_on_dataset(df,model,compute_predictions=True)
        df['CNN_SCORE'] = predictions.astype(float)
        
        predictions_df.append(df)

    df_predictions_final = pd.concat(predictions_df)

    if save_predictions_path != '':
        df_predictions_final.to_csv(save_predictions_path,index=False)
    return df_predictions_final

def evaluate_model_on_dataset(df_ref,model,compute_predictions=True):
    
    '''
    Evaluates a trained model based on the provided dataset.
    '''

    dataset_params = {'batch_size': 512,'num_workers': 4,'pin_memory':False,'shuffle':False}
    dataset = MRIDataset(reference_table=df_ref)
    dataloader = DataLoader(dataset, **dataset_params)
    
    size = len(dataloader.dataset)
    true_labels = torch.Tensor().to(device)
    predicted_logits = torch.Tensor().to(device)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)
            X = X.view(-1,1, 100,100)
            y = y.view(-1,1)

            y_pred = model(X)
            y = y.type_as(y_pred)
            
            true_labels = torch.cat((true_labels,y),0)
            predicted_logits = torch.cat((predicted_logits,y_pred),0)
        predicted_probas = torch.sigmoid(predicted_logits)
        metrics = compute_metrics_binary(y_true = true_labels, y_pred_proba = predicted_probas, threshold = 0.5,verbose=1)
                
        if compute_predictions:
            predicted_probas = predicted_probas.cpu().detach().numpy().ravel()

    return metrics,predicted_probas
