import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append("./../model_evaluation")
from ensemble_evaluation import calculate_rocs,calculate_metrics

def prepare_ensemble_experiment_set(cognitive_predictions_path,mri_predictions_path):
    
    df_mri = prepare_mri_predictions(mri_predictions_path)
    df_cog = pd.read_csv(cognitive_predictions_path)
    df_cog_final = df_cog.query("DATASET in  ('train','test','validation')")[['SUBJECT','IMAGE_DATA_ID','DATASET','COGTEST_SCORE','DIAGNOSIS']].reset_index(drop=True)
    df_ensemble = df_mri.drop(['MACRO_GROUP'],axis=1).merge(df_cog_final,on=['SUBJECT','IMAGE_DATA_ID','DATASET'])
    return df_ensemble

def prepare_mri_predictions(mri_data_path):
    df_mri = pd.read_csv(mri_data_path)
    df_mri['RUN_ID'] = df_mri['ORIENTATION'] + '_' + df_mri['SLICE'].astype(str)
    df_mri = df_mri[['SUBJECT','IMAGE_DATA_ID','ORIENTATION','SLICE','CNN_SCORE','MACRO_GROUP','DATASET','RUN_ID']]
    df_mri['DATASET'].fillna('train_cnn',inplace=True)
    df_mri = df_mri.pivot_table(index=['SUBJECT','IMAGE_DATA_ID','DATASET','MACRO_GROUP'],values=['CNN_SCORE'],columns=['RUN_ID'])
    df_mri.columns = [x[0]+'_'+x[1].upper() for x in df_mri.columns]
    df_mri.reset_index(inplace=True)
    return df_mri

def get_experiment_sets(df_ensemble,cols_to_drop = ['SUBJECT','IMAGE_DATA_ID','DATASET']):
    df_train = df_ensemble.query("DATASET == 'train'").drop(cols_to_drop,axis=1).fillna(0)
    df_validation = df_ensemble.query("DATASET == 'validation'").drop(cols_to_drop,axis=1).fillna(0)
    df_test = df_ensemble.query("DATASET == 'test'").drop(cols_to_drop,axis=1).fillna(0)
    return df_train,df_validation,df_test

def train_ensemble_models(df_train,label,models):
    trained_models = []
    for model in models:
        model.fit(df_train.drop(label,axis=1),df_train[label]);
        trained_models.append(model)
    return trained_models

def calculate_experiment_performance(models,datasets,label):
    df_rocs = calculate_rocs(models,datasets,label)
    calculate_metrics(models,datasets,df_rocs,label)
    return df_rocs

class DummyModel():
    def __init__(self,slice,threshold=0.5):
        self.slice = slice
        self.threshold = threshold

    def fit(self,X,y):
        pass
    
    def predict(self,X,y=None):
        x = X[self.slice].copy()
        x[x >= self.threshold] = 1
        x[x < self.threshold] = 0

    def predict_proba(self,X,y=None):
        x = X[self.slice].copy()
        return np.array([1-x,x]).T

class CNNCoronal(DummyModel): pass
class CNNAxial(DummyModel): pass
class CNNSagittal(DummyModel): pass
class CNN3Slices(DummyModel): pass
class CNN3SlicesCogScore(DummyModel): pass
class CNN3SlicesDemographics(DummyModel): pass
class CDRSB(DummyModel): pass
