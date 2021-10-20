# %%
import numpy as np
import pandas as pd

from pycaret.classification import *
from sklearn.metrics import fbeta_score,make_scorer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# from interpret.glassbox import ExplainableBoostingClassifier
# from pycaret.utils import enable_colab
# enable_colab()
import sys
sys.path.append("./../utils")
from utils import *
# %%
def run_tabular_data_experiment(cognitive_tests_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                                ensemble_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
                                experiment_name = 'ADNI_CN_AD',
                                labels = [0,1],
                                label_column = 'DIAGNOSIS',
                                n_splits = 5,
                                selected_models = ['lr','svm','lightgbm','et']
                                ):
    '''
        Train and validates models with tabular data, that is, cognitive tests and patients demographics.

        Parameters
        ----------

        df: Reference dataframe containing information about patients.

        n_splits: number to determine the amount of fold splits in data.

        labels: Label of the classes.

        label_column: Column containing the label class to filter the final train and test set.

        return_indices: Flag to return the train/test indices. If False, it returns the entire reference dataframes.

        Returns
        ----------
        df_results: results for the experiment.
        best_model: best performing model. 

    '''

    print("Loading dataset...")

    df_adni_merge = pd.read_csv(cognitive_tests_data_path).dropna()
    df_ensemble = pd.read_csv(ensemble_data_path).query("CONFLICT_DIAGNOSIS == False")
    df_adni_merge = df_adni_merge.merge(df_ensemble[['SUBJECT','IMAGEUID','DATASET']],on=['SUBJECT','IMAGEUID'],how='left')

    df_train = df_adni_merge.query("DATASET not in ('validation','test') and DIAGNOSIS in @labels").drop(['RID','IMAGEUID','VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT',"SUBJECT",'DIAGNOSIS_BASELINE','DATASET'],axis=1)
    df_validation = df_adni_merge.query("DATASET == 'validation' and DIAGNOSIS in @labels").drop(['RID','IMAGEUID','VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT',"SUBJECT",'DIAGNOSIS_BASELINE','DATASET'],axis=1)
    df_test = df_adni_merge.query("DATASET == 'test' and DIAGNOSIS in @labels").drop(['RID','IMAGEUID','VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT',"SUBJECT",'DIAGNOSIS_BASELINE','DATASET'],axis=1)

    print("Setting up experiment...")
    base_experiment_params = {
    'categorical_features': ['MALE','HISPANIC','RACE_WHITE', 'RACE_BLACK', 'RACE_ASIAN','MARRIED', 'WIDOWED', 'DIVORCED', 'NEVER_MARRIED'],
    'numeric_features': ['AGE','YEARS_EDUCATION','CDRSB', 'ADAS11','ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning','RAVLT_forgetting', 'RAVLT_perc_forgetting', 'TRABSCOR', 'FAQ', 'MOCA'],
    'target' : label_column,
    'transformation':True,
    'remove_multicollinearity' : False,
    'session_id':1,
    'silent':True,
    'verbose':1
    }

    base_experiment_params['data'] = df_train
    base_experiment_params['test_data'] = df_validation
    base_experiment_params['fold_strategy'] = 'stratifiedkfold'
    base_experiment_params['fold'] = 5
    base_experiment_params['experiment_name'] = experiment_name
    exp_clinical = setup(**base_experiment_params)


    print("Training models...")
    trained_models = compare_models(include=selected_models,sort='AUC',n_select = 5,turbo=True,cross_validation = True,verbose=1)
    df_validation_results = pull()

    df_test_results = compute_results(df_test,trained_models)
    df_validation_results = compute_results(df_validation,trained_models)

    model = trained_models[0]

    # TODO: set in experiment setup to ignore columns SUBJECT,DATASET and IMAGEUID without removing them from dataset, we need it to join with other info.
    df_predictions_train = pred = predict_model(model,data=df_train,verbose=0,raw_score=True)
    df_predictions_validation = pred = predict_model(model,data=df_train,verbose=0,raw_score=True)
    df_predictions_test = pred = predict_model(model,data=df_train,verbose=0,raw_score=True)
    df_predictions = pd.concat([df_predictions_train,df_predictions_validation,df_predictions_test])
    df_predictions['TABULAR_MODEL'] = type(model).__name__

    if model_path is not None and model_path != '': 
        pass
        # TODO:save model with pycaret or similar
    
    if output_path is not None and output_path != '':
        df_predictions.to_csv(output_path,index=False)

    return df_validation_results,df_test_results,df_predictions,model

def compute_results(df,trained_models):
        
    results = []
    for model in trained_models:
        pred = predict_model(model,data=df,verbose=0,raw_score=True);
        y_predict_proba = pred['Score_1'] if 'Score_1' in pred.columns else pred['Label']
        metrics = compute_metrics_binary(y_true=pred['DIAGNOSIS'], y_pred_proba = y_predict_proba,threshold = 0.5,verbose=0)
        metrics['model'] = type(model).__name__
        results.append(metrics)
    df_results = pd.DataFrame(results).sort_values('auc',ascending=False)
    return df_results


def compute_metrics_binary(y_true, y_pred_proba = None,threshold = 0.5,verbose=0):
    
    y_pred_label = y_pred_proba
    y_pred_label[y_pred_proba >= threshold] = 1
    y_pred_label[y_pred_proba < threshold] = 0
    
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

# %%

cognitive_tests_data_path = './../../data/COGNITIVE_DATA_PREPROCESSED.csv'
ensemble_data_path = './../../data/PROCESSED_ENSEMBLE_REFERENCE.csv'
output_path = ''
model_path = ''
experiment_name = 'ADNI_CN_AD'
labels = [0,1]
label_column = 'DIAGNOSIS'
n_splits = 5
selected_models = ['lr','lightgbm','et']

df_validation_results,df_test_results,df_predictions = run_tabular_data_experiment(
                                                                                        cognitive_tests_data_path=cognitive_tests_data_path,
                                                                                        ensemble_data_path=ensemble_data_path,
                                                                                        model_path = model_path,
                                                                                        output_path = output_path,
                                                                                        experiment_name=experiment_name,
                                                                                        labels=labels,
                                                                                        n_splits=n_splits,
                                                                                        selected_models=selected_models)
# %%
