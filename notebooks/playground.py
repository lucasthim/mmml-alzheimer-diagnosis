# %%
import os
import pandas as pd
import numpy as np

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/src/model_training/')
from ensemble_train import DummyModel

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/src/model_evaluation/')
from base_evaluation import find_optimal_cutoff
from ensemble_evaluation import calculate_metrics_on_datasets,calculate_rocs_on_datasets

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/')


# %%
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def find_optimal_threshold(predict_proba,label):
    ''' 
    Finds an optimal threshold point based on the ROC curve. 
    The optimal point is the one closest to the upper left corner of the curve. 
    Ideally, that point would be the vertice itself if the area under the ROC curve was one.
    
    Parameters
    ----------

    predict_proba: list of predicted scores

    label: list of true labels
    '''
    fpr, tpr, thresholds = roc_curve(label, predict_proba, drop_intermediate=False)
    _,_,opt_threshold = find_optimal_cutoff(fpr, tpr, thresholds)
    print("Optimal threshold: % .4f" % opt_threshold)
    return opt_threshold


# %%
orientation = 'coronal'
df_mci = pd.read_csv('./data/PREDICTIONS_MCI_VGG19_BN_1125.csv')
df_orientation = df_mci.query("ORIENTATION == @orientation")
df_train = df_orientation.query("DATASET not in ('test','validation')")
df_validation = df_orientation.query("DATASET  in ('','validation')")
df_test = df_orientation.query("DATASET  in ('test','')")

# %%

df = df_train
label = 'MACRO_GROUP'
predict_proba = 'CNN_SCORE'
opt_threshold = find_optimal_threshold(df[predict_proba],df[label])
# %%
models = [DummyModel(slice='CNN_SCORE')]
datasets=[df_train,df_validation,df_test]

df_rocs_cnns = calculate_rocs_on_datasets(models=models,datasets=datasets,label='MACRO_GROUP')
calculate_metrics_on_datasets(models,datasets,df_rocs_cnns,label)
# %%
#############################################################
#############################################################


