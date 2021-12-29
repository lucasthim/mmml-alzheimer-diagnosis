import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from base_evaluation import *


def compare_ensembles_performance_on_dataset(dataset,df_rocs,label,model_names):
    '''
    Compare the performance of several ensemble models against each other.
    Plots comparison bar plots for AUC,Accuracy,F1,Recall and Precision.

    Parameters
    ----------

    df: dataset to compare performance of models


    '''

    pass

def compare_ensembles_rocs_on_dataset(df:pd.DataFrame,label:str,model_names:list):
    '''
    Compare ROC curves of several predictors by plotting the curves and their respective AUCs.
    

    Parameters
    ------------

    df: DataFrame containing columns with predicted scores and true label

    label: string indicating which column of the dataframe is the true label.

    model_names: list of strings indicating the predicted score columns of the dataframe.

    Returns
    ------------

    Returns the AUC and optimal thresholds for each model.

    '''
    fig =plt.figure(figsize=(8,8))
    y_true = df[label].values
    df_results = pd.DataFrame(columns=['AUC','Optimal_Threshold'])
    for model in model_names:
    
      y_proba = df[model]
      fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba, drop_intermediate=False)
      auc = metrics.auc(fpr, tpr)
      df_results.loc[model,'AUC'] = auc
      _,_,df_results.loc[model,'Optimal_Threshold'] = find_optimal_cutoff(fpr, tpr, thresholds)
    
      plt.plot(fpr, tpr, label=model + '(AUC = %.4f'%auc + ')')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return df_results

def calculate_experiment_performance_on_datasets(models,datasets,label):
    
    '''
    Calculate the performance of an experiment by calculating the ROC and some other metrics.

    Parameters
    ----------

    models: array with trained model objects of the current experiment.

    datasets: train, validation and test datasets used in the experiment. 
    
    label: Label column in the datasets.
    '''
    
    df_rocs = calculate_rocs_on_datasets(models=models,datasets=datasets,label=label)
    calculate_metrics_on_datasets(models,datasets,df_rocs,label)
    return df_rocs

def calculate_rocs_on_datasets(models:list,datasets:list,dataset_names:list=['Train','Validation','Test'],label:str ='MACRO_GROUP'):
    '''
    Function that calculates the ROC curve along with some statistics
    for a list of given models for Train,Validation and Test.

    Parameters
    ----------

    models: list or array of model objects following the sklearn pattern.

    datasets: list or array containing the Train, Validation and Test sets.

    label: indication of which column of the dataset has the true label.
    '''
    
    dfs=[]
    for set,df in zip(dataset_names,datasets):

      df_roc,_ = calculate_and_plot_roc(df, models=models,levels=[0.75, 0.9], label=label,set=set)
      df_roc['set'] = set
      dfs.append(df_roc)
      print('')
    df_rocs = pd.concat(dfs).reset_index()
    df_rocs = set_threshold_for_test(df_rocs,models,reference='Validation')
    return df_rocs

def set_threshold_for_test(df_rocs,models,reference='Validation'):
  
  '''
  Set the optimal threshold value for test set based on the validation or train threshold.
  '''

  for model in models:
    model_name = type(model).__name__ if not(isinstance(model,str)) else model
    reference_threshold = df_rocs.query("index == @model_name and set == @reference").iloc[0]['Optimal_Thresh']
    df_rocs.loc[(df_rocs['set'] == 'Test') & (df_rocs['index'] == model_name),'Optimal_Thresh'] = reference_threshold
  return df_rocs

def calculate_metrics_on_datasets(models:list,datasets:list,df_rocs:pd.DataFrame,label:str):

    for set,df in zip(['Train','Validation','Test'],datasets):
        for model in models:
            model_name = type(model).__name__
            optimal_threshold = df_rocs.query("index== @model_name and set == @set")['Optimal_Thresh'].values[0]

            print(f"{model_name} Results for {set}:")
            print("Optimal Threshold: %.4f" % optimal_threshold)
            
            y_true = df[label]
            y_pred_proba = model.predict_proba(df.drop(label,axis=1))[:,-1]
            result_metrics = compute_metrics_binary(y_true, y_pred_proba = y_pred_proba,threshold = optimal_threshold,verbose=1)
        print("\n---------------------------------------")