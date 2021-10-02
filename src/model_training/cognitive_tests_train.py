# %%
import numpy as np
import pandas as pd

from pycaret.classification import *
from sklearn.metrics import fbeta_score,make_scorer

from interpret.glassbox import ExplainableBoostingClassifier
# from pycaret.utils import enable_colab
# enable_colab()
import sys
sys.path.append("./../utils")
from utils import *
# %%
def run_tabular_data_experiment(cognitive_tests_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/COGNITIVE_DATA_PREPROCESSED.csv',
                                ensemble_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/ENSEMBLE_REFERENCE.csv',
                                experiment_name = 'ADNI_CN_AD',
                                labels = [0,1],
                                label_column = 'DIAGNOSIS',
                                n_splits = 5,
                                selected_models = ['lr','svm','lightgbm','et',ExplainableBoostingClassifier()]
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
        df_validation_results: results for the cross validation
        df_test_results: results for the test set
        trained_models: trained models over the training set

    '''
    
    print("Loading dataset...")

    df_adni_merge = pd.read_csv(cognitive_tests_data_path).dropna()
    df_ensemble = pd.read_csv(ensemble_data_path)

    print("Setting up experiment...")
    
    base_experiment_params = {
    'categorical_features': ['MALE','HISPANIC','RACE_WHITE', 'RACE_BLACK', 'RACE_ASIAN','MARRIED', 'WIDOWED', 'DIVORCED', 'NEVER_MARRIED'],
    'numeric_features': ['AGE','YEARS_EDUCATION','CDRSB', 'ADAS11','ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning','RAVLT_forgetting', 'RAVLT_perc_forgetting', 'TRABSCOR', 'FAQ', 'MOCA'],
    'target' : label_column,
    'transformation':True,
    'remove_multicollinearity' : False,
    'session_id':1,
    'silent':True,
    'verbose':0
    }

    # TODO: import ENSEMBLE_REFERENCE and filter columns DATASET to get train,val and test sets.
    # TODO: remove cross validation, unless its just for hyperparams tunning.
    # WARNING: Validation and test sets should be the same as ensemble. Train set can contain more samples.
    
    df_train, df_test = train_test_split_by_subject(df_adni_merge,test_size=0.2,label_column=label_column,labels=labels)
    base_experiment_params['data'] = df_train.drop(["SUBJECT",'DIAGNOSIS_BASELINE'],axis=1)
    base_experiment_params['test_data'] = df_test.drop(["SUBJECT",'DIAGNOSIS_BASELINE'],axis=1)
    base_experiment_params['fold_strategy'] = StratifiedSubjectKFold(df_train,labels=labels,n_splits=n_splits,label_column=label_column)
    base_experiment_params['experiment_name'] = experiment_name
    exp_clinical = setup(**base_experiment_params)

    print("Training models...")
    trained_models = compare_models(include=selected_models,sort='AUC',n_select = 5,turbo=True,cross_validation = True,verbose=0)
    print("Models trained and validated!")
    print('-----------------------------------------------------')
    df_validation_results = pull().drop(['Kappa','MCC'],axis=1)
    print("Validation results: \n",df_validation_results)
    print('-----------------------------------------------------')
    df_test_results = []
    for model in trained_models:
    predict_model(model,verbose=0);
    test_performance = pull();
    df_test_results.append(test_performance)
    df_test_results = pd.concat(df_test_results).reset_index(drop=True).drop(['Kappa','MCC'],axis=1).sort_values('AUC',ascending=False)

    print("Test results: \n",df_test_results)
    print('-----------------------------------------------------')

    return df_validation_results,df_test_results,trained_models


def prepare_dataset_for_training(path):
    pass