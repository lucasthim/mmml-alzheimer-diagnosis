import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression

def prepare_mri_predictions(mri_data_path):
    df_mri = pd.read_csv(mri_data_path)
    df_mri['RUN_ID'] = df_mri['ORIENTATION'] + '_' + df_mri['SLICE'].astype(str)
    df_mri = df_mri[['SUBJECT','IMAGE_DATA_ID','ORIENTATION','SLICE','CNN_SCORE','MACRO_GROUP','DATASET','RUN_ID']]
    df_mri['DATASET'].fillna('train_cnn',inplace=True)
    df_mri = df_mri.pivot_table(index=['SUBJECT','IMAGE_DATA_ID','DATASET','MACRO_GROUP'],values=['CNN_SCORE'],columns=['RUN_ID'])
    df_mri.columns = [x[0]+'_'+x[1].upper() for x in df_mri.columns]
    df_mri.reset_index(inplace=True)
    return df_mri

def prepare_ensemble_sets(df_cog,df_mri):
    # Merging predictions
    df_cog_final = df_cog.query("DATASET in  ('train','test','validation')")[['SUBJECT','IMAGE_DATA_ID','DATASET','COGTEST_SCORE','DIAGNOSIS']].reset_index(drop=True)
    df_ensemble = df_mri.drop(['MACRO_GROUP'],axis=1).merge(df_cog_final,on=['SUBJECT','IMAGE_DATA_ID','DATASET'])
    return df_ensemble


class DummyModel():
    def __init__(self,slice,threshold=0.5):
        self.slice = slice
        self.threshold = threshold

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
