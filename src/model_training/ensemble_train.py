# TODO: 
# 1.read MRI predictions, 
# 2. read tabular data predictions, 
# 3. merge both dataframes
# 3.5 (Optional) Select wich categories to ensemble (coronal, axial, sagittal and tabular)
# 4. select ensemble type = EBM, mean value
# 5. train, validate and test ensemble.
# 6. generate final predictions and concat with previous predictions
# Compute overall performance for all categories: CNN, Tabular, Ensemble


# %%
import sys
import numpy as np
import pandas as pd
from pycaret.classification import *

import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score,make_scorer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from interpret.glassbox import ExplainableBoostingClassifier
# from pycaret.utils import enable_colab
# enable_colab()
from cognitive_tests_train import *
sys.path.append("./../utils")
from utils import *

# %%
def run_tabular_data_experiment(cognitive_tests_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                                mri_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
                                experiment_name = 'ADNI_CN_AD',
                                labels = [0,1],
                                label_column = 'DIAGNOSIS',
                                n_splits = 5,
                                selected_models = ['lr','lightgbm',ExplainableBoostingClassifier()],
                                model_path = '',
                                output_path = ''
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
    df_ensemble = pd.read_csv(mri_data_path).query("CONFLICT_DIAGNOSIS == False")
    df_adni_merge = df_adni_merge.merge(df_ensemble[['SUBJECT','IMAGEUID','DATASET']],on=['SUBJECT','IMAGEUID'],how='left')

    df_train = df_adni_merge.query("DATASET not in ('validation','test') and DIAGNOSIS in @labels").drop(['VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT','RACE','DIAGNOSIS_BASELINE'],axis=1)
    df_validation = df_adni_merge.query("DATASET == 'validation' and DIAGNOSIS in @labels").drop(['VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT','RACE','DIAGNOSIS_BASELINE'],axis=1)
    df_test = df_adni_merge.query("DATASET == 'test' and DIAGNOSIS in @labels").drop(['VISCODE','SITE','COLPROT','EXAMDATE','ORIGPROT','RACE','DIAGNOSIS_BASELINE'],axis=1)

    print("Setting up experiment...")
    base_experiment_params = {
    'categorical_features': ['MALE','HISPANIC','RACE_WHITE', 'RACE_BLACK', 'RACE_ASIAN','MARRIED', 'WIDOWED', 'DIVORCED', 'NEVER_MARRIED'],
    'numeric_features': ['AGE','YEARS_EDUCATION','CDRSB', 'ADAS11','ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning','RAVLT_forgetting', 'RAVLT_perc_forgetting', 'TRABSCOR', 'FAQ', 'MOCA'],
    'ignore_features':['RID','SUBJECT','IMAGEUID','DATASET'],
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

    print(f"Computing results for validation set ({df_validation.shape})... ")
    df_validation_results = compute_results(df_validation,trained_models)
    print(f"Computing results for test set ({df_test.shape})... ")
    df_test_results = compute_results(df_test,trained_models)

    model = trained_models[0]
    print(f"Saving predictions for best model: {type(model).__name__}")
    # TODO: set in experiment setup to ignore columns SUBJECT,DATASET and IMAGEUID without removing them from dataset, we need it to join with other info.
    df_predictions_train = predict_model(model,data=df_train,verbose=0,raw_score=True)
    df_predictions_validation = predict_model(model,data=df_validation,verbose=0,raw_score=True)
    df_predictions_test = predict_model(model,data=df_test,verbose=0,raw_score=True)
    df_predictions = pd.concat([df_predictions_train,df_predictions_validation,df_predictions_test])
    df_predictions['TABULAR_MODEL'] = type(model).__name__

    if model_path is not None and model_path != '': 
        pass
        # TODO:save model with pycaret or similar
    
    if output_path is not None and output_path != '':
        df_predictions.to_csv(output_path,index=False)

    return df_validation_results,df_test_results,df_predictions,model

def run_ensemble_experiment(cognitive_tests_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                                mri_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
                                experiment_name = 'ADNI_CN_AD',
                                labels = [0,1],
                                label_column = 'DIAGNOSIS',
                                n_splits = 5,
                                selected_models = ['lr','svm','lightgbm','et'],
                                model_path = '',
                                output_path = ''
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

    df_mri = pd.read_csv(mri_data_path)
    df_mri['CNN_LABEL'].replace({False:0,True:1},inplace=True)
    df_mri = df_mri.query("ROTATION_ANGLE == 0 and DATASET in ('train','test','validation')")[['SUBJECT','IMAGE_DATA_ID','ORIENTATION','SLICE','RUN_ID','CNN_LABEL','CNN_SCORE','MACRO_GROUP','DATASET']]
    df_mri = df_mri.pivot_table(index=['SUBJECT','IMAGE_DATA_ID','DATASET','MACRO_GROUP'],values=['CNN_LABEL','CNN_SCORE'],columns=['RUN_ID'])
    df_mri.columns = [x[0]+'_'+x[1].upper() for x in df_mri.columns]
    df_mri.reset_index(inplace=True)
    df_mri.drop(["CNN_LABEL_AXIAL23",'CNN_LABEL_CORONAL43','CNN_LABEL_SAGITTAL26','MACRO_GROUP'],axis=1,inplace=True)

    df_cog = pd.read_csv(cognitive_tests_data_path)
    df_cog.rename(columns={"IMAGEUID":"IMAGE_DATA_ID"},inplace=True)
    df_cog = df_cog.query("DATASET in  ('train','test','validation')")[['SUBJECT','IMAGE_DATA_ID','DATASET','Score_1','DIAGNOSIS']].reset_index(drop=True)
    df_cog['IMAGE_DATA_ID'] = 'I' + df_cog['IMAGE_DATA_ID'].astype(str)
    df_ensemble = df_mri.merge(df_cog,on=['SUBJECT','IMAGE_DATA_ID','DATASET'])

    df_train = df_ensemble.query("DATASET not in ('train') and DIAGNOSIS in @labels")
    df_validation = df_ensemble.query("DATASET == 'validation' and DIAGNOSIS in @labels")
    df_test = df_ensemble.query("DATASET == 'test' and DIAGNOSIS in @labels")

    print("Setting up experiment...")
    base_experiment_params = {
    'numeric_features': ['CNN_SCORE_CORONAL43','CNN_SCORE_SAGITTAL26','CNN_SCORE_AXIAL23', 'Score_1'],
    'ignore_features':['SUBJECT','IMAGE_DATA_ID','DATASET'],
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

    print(f"Computing results for validation set ({df_validation.shape})... ")
    df_validation_results = compute_results(df_validation,trained_models)
    
    print(f"Computing results for test set ({df_test.shape})... ")
    df_test_results = compute_results(df_test,trained_models)

    model = trained_models[0]
    print(f"Saving predictions for best model: {type(model).__name__}")

    df_predictions_train = predict_model(model,data=df_train,verbose=0,raw_score=True)
    df_predictions_validation = predict_model(model,data=df_validation,verbose=0,raw_score=True)
    df_predictions_test = predict_model(model,data=df_test,verbose=0,raw_score=True)
    df_predictions = pd.concat([df_predictions_train,df_predictions_validation,df_predictions_test])
    df_predictions['TABULAR_MODEL'] = type(model).__name__

    if model_path is not None and model_path != '': 
        pass
        # TODO:save model with pycaret or similar
    
    if output_path is not None and output_path != '':
        df_predictions.to_csv(output_path,index=False)

    return df_validation_results,df_test_results,df_predictions,model

def preprocess_cnn_predictions(df):
    df_mri = df.copy()
    df_mri['CNN_LABEL'].replace({False:0,True:1},inplace=True)
    df_mri['DATASET'].fillna('CNN_train',inplace=True)
    df_mri = df_mri[['SUBJECT','IMAGE_DATA_ID','ORIENTATION','SLICE','RUN_ID','CNN_LABEL','CNN_SCORE','MACRO_GROUP','DATASET']]
    df_mri = df_mri.pivot_table(index=['SUBJECT','IMAGE_DATA_ID','DATASET','MACRO_GROUP'],values=['CNN_LABEL','CNN_SCORE'],columns=['RUN_ID'])
    df_mri.columns = [x[0]+'_'+x[1].upper() for x in df_mri.columns]
    df_mri.reset_index(inplace=True)
    df_mri.drop(["CNN_LABEL_AXIAL8",'CNN_LABEL_CORONAL70','CNN_LABEL_SAGITTAL50'],axis=1,inplace=True)
    for col in ['CNN_SCORE_AXIAL8','CNN_SCORE_CORONAL70','CNN_SCORE_SAGITTAL50']:
    nulls = df_mri[col].isna()
    if nulls.sum() > 0:
        print(f"Found {nulls.sum()} null predictions. Replacing by 0.")
        df_mri.loc[nulls,col] = 0
    else:
        print(f"No null predictions for {col}")
    return df_mri

# %%

cognitive_tests_data_path = './../../data/COGNITIVE_DATA_PREPROCESSED.csv'
ensemble_data_path = './../../data/PROCESSED_ENSEMBLE_REFERENCE.csv'
output_path = ''
model_path = ''
experiment_name = 'ADNI_CN_AD'
labels = [0,1]
label_column = 'DIAGNOSIS'
n_splits = 5
selected_models = ['lightgbm','lr'] #et,ebm

df_validation_results,df_test_results,df_predictions,model = run_tabular_data_experiment(
                                                                                        cognitive_tests_data_path=cognitive_tests_data_path,
                                                                                        ensemble_data_path=ensemble_data_path,
                                                                                        model_path = model_path,
                                                                                        output_path = output_path,
                                                                                        experiment_name=experiment_name,
                                                                                        labels=labels,
                                                                                        n_splits=n_splits,
                                                                                        selected_models=selected_models)
# %%
df_ensemble = pd.read_csv(cognitive_tests_data_path)
df_ensemble
# %%
