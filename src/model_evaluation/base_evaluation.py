import math
from itertools import combinations

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erfcinv
from sklearn.metrics import roc_auc_score,roc_curve,auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from de_long_evaluation import delong_roc_test

def compute_metrics_binary(y_true, y_pred_proba,threshold = 0.5,verbose=0):
    '''
    
    Compute the following metrics for a binary classification problem: 
    AUC, Accuracy, F1 Score, Precision, Recall and Confusion matrix.

    Parameters
    ----------
    y_true: list or array containing the true values.

    y_pred_proba: array containing the predicted scores

    threshold: cutoff point to encode the predicted scores. 1 if score >= threshold, else 0.

    verbose: Flag to print out results.

    Returns
    ---------

    Dict with metrics and their values.

    '''
    
    y_pred_proba = get_numpy_array(y_pred_proba)
    y_pred_label = y_pred_proba.copy()
    y_pred_label[y_pred_proba >= threshold] = 1
    y_pred_label[y_pred_proba < threshold] = 0
    
    y_true = get_numpy_array(y_true)

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

def calculate_and_plot_roc(df, models, levels=[0.75,0.9],label='DIAGNOSIS',set='Train',title_prefix=''):
    
    '''
    Function that calculates and plots the ROC Curve along with some statistics
    for a list of given models.

    Parameters
    ----------
    df: dataset to calculate ROC and statistics.

    models: array containing the trained models or strings indicating model names.

    levels: levels to measure sensitivity and other statistics.

    label: indication of which column of the dataset has the true label.
    
    set: name of the dataset

    Returns
    --------
    '''
    roc_df = pd.DataFrame(columns=[f'SensLevel_at_{levels[0]}', f'SensLevel_at_{levels[1]}',
                                   'AUC', 'AUC_CI_low', 'AUC_CI_high',
                                   'Std_Error',
                                   'Optimal_Sen', 'Sen_CI_low', 'Sen_CI_high',
                                   'Optimal_Spe', 'Spe_CI_low', 'Spe_CI_high'])
    true_labels = df[label]

    fig =plt.figure(figsize=(8,8))

    for model in models:

        if isinstance(model,str):
            y_proba = df[model]
            model_name = model
        else:
            y_proba = model.predict_proba(df.drop(label,axis=1))[:,-1]
            model_name = type(model).__name__
        roc_df.loc[model_name,'Model'] = model
        # Compute False postive rate, and True positive rate
        fpr, tpr, thresholds = roc_curve(true_labels, y_proba, drop_intermediate=False)
        # Calculate Area under the curve to display on the plot
        # auc = roc_auc_score(y_test, model.predict(x_test))
        roc_df.loc[model_name, 'AUC'] = auc(fpr, tpr)

        # calculate the sensitivity at levels
        roc_df.loc[model_name, f'SensLevel_at_{levels[0]}'] = calculate_sensibility_at_level(tpr, fpr, levels[0])
        roc_df.loc[model_name, f'SensLevel_at_{levels[1]}'] = calculate_sensibility_at_level(tpr, fpr, levels[1])

        # Calculate the standard error of AUC
        roc_df.loc[model_name, 'Std_Error'] = calculate_std_error_auc(roc_df.loc[model_name, 'AUC'], true_labels)

        # Calculate the confidence interval of AUC
        roc_df.loc[model_name, ['AUC_CI_low', 'AUC_CI_high']] = calculate_confidence_interval_auc(roc_df.loc[model_name, 'AUC'], 
                                                                                                  roc_df.loc[model_name, 'Std_Error'])

        # Calculate the optimal cutoff point, Sensitivity and specificity
        roc_df.loc[model_name, 'Optimal_Sen'], roc_df.loc[model_name, 'Optimal_Spe'], roc_df.loc[model_name, 'Optimal_Thresh'] = find_optimal_cutoff(fpr, tpr, thresholds)

        # Calculate the confidence interval of Sensitivity
        roc_df.loc[model_name, ['Sen_CI_low', 'Sen_CI_high']] = calculate_confidence_interval_sensitivity(roc_df.loc[model_name, 'Optimal_Sen'], true_labels)

        # Calculate the confidence interval of Specificity
        roc_df.loc[model_name, ['Spe_CI_low', 'Spe_CI_high']] = calculate_confidence_interval_specificity(roc_df.loc[model_name, 'Optimal_Spe'], true_labels)

        # Plot the computed values
        plt.plot(fpr, tpr, label=model_name + ' (AUC = %.3f'%roc_df.loc[model_name, 'AUC'] + ')')
    # Custom settings for the plot
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title(title_prefix  + f'{set} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()  # Display
    #plt.savefig(filename, format='png', dpi=300)
    return roc_df, fig

def calculate_sensibility_at_level(tpr, fpr, level):
    level_fpr = 1 - level  # fpr is (1-specificity)
    f_sens = interp1d(fpr, tpr)  # interpolate sensibility (tpr = sensibility)
    return (f_sens(level_fpr))

def find_optimal_cutoff(fpr, tpr, thresholds):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate.
    
    Parameters
    ----------
    fpr: False positive rate

    tpr : True positive rate

    Returns
    -------
    cutoff value

    """
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))  # Minimum distance to the upper left corner (By Pathagoras' theorem)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    return optimal_sensitivity, optimal_specificity, optimal_threshold

def calculate_std_error_auc(auc, cls):
    """
    Standard error of area under the curve
    :param auc: area under the curve
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: standard error
    """
    auc2 = auc ** 2
    q1 = auc / (2 - auc)
    q2 = 2 * auc2 / (1 + auc)
    lu = sum(cls == 1)  # Number of unhealthy subjects (class == 1)
    lh = sum(cls == 0)  # Number of healthy subjects (class == 0)
    V = (auc * (1 - auc) + (lu - 1) * (q1 - auc2) + (lh - 1) * (q2 - auc2)) / (lu * lh)
    se = math.sqrt(V)
    return se

def calculate_confidence_interval_auc(auc, std, alpha=0.05):
    """
    Confidence interval of AUC
    :param auc: area under the curve
    :param std: standard error
    :param alpha: significance level (default = 0.05)
    :return: confidence interval
    """
    ci_lo = auc + (-1 * math.sqrt(2) * erfcinv(alpha) * std)
    ci_up = auc + (math.sqrt(2) * erfcinv(alpha) * std)
    return ci_lo, ci_up

def calculate_confidence_interval_sensitivity(optimal_sensitivity, cls):
    """
    Confidence interval of Sensitivity using Simple Asymptotic
    :param optimal_sensitivity: optimal cutoff point
    :param cls: the column of the tag: unhealthy (1) and healthy (0)
    :return: confidence interval - array[(low, high)]
    """
    num_u = sum(cls == 1) 
    sa = 1.96 * math.sqrt(optimal_sensitivity * (1 - optimal_sensitivity) / num_u)
    ci_sensitivity = np.zeros(2)
    ci_sensitivity = [optimal_sensitivity - sa, optimal_sensitivity + sa]
    return ci_sensitivity

def calculate_confidence_interval_specificity(optimal_specificity, cls):
    """
        Confidence interval of Specificity using Simple Asymptotic
        :param optimal_specificity: optimal cutoff point
        :param cls: the column of the tag: unhealthy (1) and healthy (0)
        :return: confidence interval - array[(low, high)]
    """
    num_h = sum(cls == 0)  # Number of healthy subjects (class == 0)
    sa = 1.96 * math.sqrt(optimal_specificity * (1 - optimal_specificity) / num_h)
    ci_spe = np.zeros(2)
    ci_spe = [optimal_specificity - sa, optimal_specificity + sa]
    return ci_spe

def get_numpy_array(arr):
    if isinstance(arr,torch.Tensor):
        return arr.cpu().detach().numpy()
    elif isinstance(arr,list):
        return np.array(arr)
    return arr

def check_auc_difference(models,datasets,label='MACRO_GROUP',alpha=0.05):
    for model1,model2 in combinations(models,2):
        print(f"Comparing AUCs between {model1} and {model2}:")
        for set,df in zip(['Validation','Test'],datasets[1:]):
            
            log10_pvalue = delong_roc_test(df[label], df[model1], df[model2])
            pvalue = 10** log10_pvalue
            print(f"set: {set}")
            print(" p-value = %.4f" % pvalue)

            ci = int((1 - alpha) * 100)
            if pvalue <= alpha:
                print(f" Refect null hypothesis: AUCs are statistically different with {ci}% confidence.")
            else:
                print(" Cannot reject null hypothesis. AUCs are statistically the same.")

            print("")
        print("------------------------------------------")