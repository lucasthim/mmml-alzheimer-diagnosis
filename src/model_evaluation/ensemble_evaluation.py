import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from base_evaluation import *

def compare_rocs(df,label,models):
    '''
    Compare ROC curves of several predictors by plotting the curves and their respective AUCs.

    Parameters
    ------------

    df: DataFrame containing columns with predicted scores and true label

    label: string indicating which column of the dataframe is the true label.

    models: list of strings indicating the predicted score columns of the dataframe.

    Returns
    ------------

    Returns the AUC and optimal thresholds for each model.

    '''
    fig =plt.figure(figsize=(8,8))
    y_true = df[label].values
    opt_thresholds = pd.DataFrame(columns=['AUC','Optimal_Threshold'])
    for model in models:
      
        y_proba = df[model]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_proba, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        opt_thresholds.loc[model,'AUC'] = auc
        opt_thresholds.loc[model,'Optimal_Threshold'] = find_optimal_cutoff(fpr, tpr, thresholds)
      
        plt.plot(fpr, tpr, label=model + '(AUC = %.4f'%auc + ')')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def calculate_rocs(models,datasets,label):
  dfs=[]
  for set,df in zip(['Train','Validation','Test'],datasets):

    df_roc,_ = plotroc(df, models=models,levels=[0.75, 0.9], label=label,set=set)
    df_roc['set'] = set
    dfs.append(df_roc)
    print('')
  df_rocs = pd.concat(dfs)
  return df_rocs.reset_index()

def calculate_metrics(models,datasets,df_rocs,label):

  for set,df in zip(['Train','Validation','Test'],datasets):

    for model in models:
      model_name = type(model).__name__
      if set == 'Train':
        optimal_threshold = df_rocs.query("index== @model_name and set == 'Train'")['Optimal_Thresh'].values[0]
      else:
        optimal_threshold = df_rocs.query("index== @model_name and set == 'Validation'")['Optimal_Thresh'].values[0]

      print(f"{model_name} Results for {set}:")
      print("Optimal Threshold: %.4f" % optimal_threshold)
      
      y_true = df[label]
      y_pred_proba = model.predict_proba(df.drop(label,axis=1))[:,-1]
      result_metrics = compute_metrics_binary(y_true, y_pred_proba = y_pred_proba,threshold = optimal_threshold,verbose=1)
    print("\n---------------------------------------")