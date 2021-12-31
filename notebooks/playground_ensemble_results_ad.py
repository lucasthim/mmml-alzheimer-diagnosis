# %%
"""# Imports"""
%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression
from interpret import show
# %matplotlib inline
plt.style.use('seaborn')
pd.set_option('display.max_rows', 100)
# %%

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/src/model_training/')
from ensemble_train import *

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/src/model_explanation/')
from ensemble_explanation import show_feature_weights

os.chdir('/home/lucas/projects/mmml-alzheimer-diagnosis/src/model_evaluation/')
from ensemble_evaluation import compare_ensembles_rocs_on_dataset,calculate_rocs_on_datasets,calculate_metrics_on_datasets,calculate_experiment_performance_on_datasets
# %%
"""# Reprocessing Cog Tests Experiment"""

def process_cognitive_data(preprocessed_cognitive_data_path,processed_ensemble_data_path,labels):
  df_cog_data = pd.read_csv(preprocessed_cognitive_data_path).dropna()
  df_ensemble_processed = pd.read_csv(processed_ensemble_data_path)

  df_cog_data.rename(columns={"IMAGEUID":"IMAGE_DATA_ID"},inplace=True)
  df_cog_data['IMAGE_DATA_ID'] = 'I' + df_cog_data['IMAGE_DATA_ID'].astype(str)
  df_cog_data = df_cog_data.merge(df_ensemble_processed[['IMAGE_DATA_ID','DATASET']],on='IMAGE_DATA_ID',how='left').query("DIAGNOSIS in @labels").reset_index(drop=True)
  
  if max(labels) == 2:
    # replace 2 (MCI) for 1 for binary classification 
    df_cog_data['DIAGNOSIS'].replace({2:1},inplace=True)
  df_cog_data.sort_values(['DATASET',"SUBJECT",'IMAGE_DATA_ID','VISCODE','RID'],inplace=True)
  return df_cog_data

def get_cognitive_data_predictions(df_cog_data):
  df_train = df_cog_data.query("DATASET not in ('validation','test')").drop(['RID','SUBJECT','VISCODE','SITE','COLPROT','ORIGPROT','EXAMDATE','IMAGE_DATA_ID','DIAGNOSIS_BASELINE','RACE','DATASET'],axis=1)
  
  ebm_cog = ExplainableBoostingClassifier();
  ebm_cog.fit(df_train.drop('DIAGNOSIS',axis=1),df_train['DIAGNOSIS']);
  
  df_cog_final = df_cog_data.drop(['SITE','COLPROT','ORIGPROT','EXAMDATE','DIAGNOSIS_BASELINE','RACE'],axis=1)
  df_cog_final['COGTEST_SCORE'] = ebm_cog.predict_proba(df_cog_final.drop(['DIAGNOSIS','DATASET',"SUBJECT",'IMAGE_DATA_ID','VISCODE','RID'],axis=1))[:,-1]
  
  return df_cog_final,ebm_cog

#%%
preprocessed_cognitive_data_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/COGNITIVE_DATA_PREPROCESSED.csv'
processed_ensemble_data_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PROCESSED_ENSEMBLE_REFERENCE.csv'
cognitive_data_predictions_final_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_AD_COG_TESTS_1125.csv'
classes = [0,1]

df_cog_data = process_cognitive_data(preprocessed_cognitive_data_path,processed_ensemble_data_path,labels=classes)
df_cog_final,ebm_cog = get_cognitive_data_predictions(df_cog_data)

# %%
label = 'DIAGNOSIS'
df_train = df_cog_final.query("DATASET not in ('validation','test')").drop(['COGTEST_SCORE','DATASET',"SUBJECT",'IMAGE_DATA_ID','VISCODE','RID'],axis=1).fillna(0)
df_validation = df_cog_final.query("DATASET in ('validation')").drop(['COGTEST_SCORE','DATASET',"SUBJECT",'IMAGE_DATA_ID','VISCODE','RID'],axis=1).fillna(0)
df_test = df_cog_final.query("DATASET in ('test')").drop(['COGTEST_SCORE','DATASET',"SUBJECT",'IMAGE_DATA_ID','VISCODE','RID'],axis=1).fillna(0)

models=[ebm_cog]
datasets=[df_train,df_validation,df_test]
df_rocs = calculate_experiment_performance_on_datasets(models,datasets,label)
df_rocs 


#%%

"""# Ensemble Experiments"""

# mri_data_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_MCI_VGG19_BN_1125.csv'
# cognitive_data_predictions_final_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_MCI_COG_TESTS_1125.csv'

mri_data_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_AD_VGG19_BN_202111252.csv'
cognitive_data_predictions_final_path = '/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_AD_COG_TESTS_1125.csv'

df_mri = prepare_mri_predictions(mri_data_path)
df_ensemble = prepare_ensemble_experiment_set(cognitive_data_predictions_final_path,mri_data_path)
# %%

"""## Experiment0 using each CNN prediction alone"""

df_preds = pd.read_csv(mri_data_path)

label = 'MACRO_GROUP'
df_train = df_mri.query("DATASET not in ('validation','test')").fillna(0)
df_validation = df_mri.query("DATASET in ('validation')").fillna(0)
df_test = df_mri.query("DATASET in ('test')").fillna(0)

models=[CNNCoronal(slice='CNN_SCORE_CORONAL_43'),
        CNNSagittal(slice='CNN_SCORE_SAGITTAL_26'),
        CNNAxial(slice='CNN_SCORE_AXIAL_23')]
datasets=[df_train,df_validation,df_test]


df_rocs_cnns = calculate_experiment_performance_on_datasets(models,datasets,label)
# df_rocs_cnns = calculate_rocs_on_datasets(models=models,datasets=datasets,label=label)
# calculate_metrics_on_datasets(models,datasets,df_rocs_cnns,label)

# %%
"""## Experiment1 using Features Coronal43, Axial8, Sagittal50, CognitiveTestScore"""

df_train,df_validation,df_test = get_experiment_sets(df_ensemble,cols_to_drop = ['SUBJECT','DATASET'])

datasets = [df_train,df_validation,df_test]
label = 'DIAGNOSIS'
ebm,lr = ExplainableBoostingClassifier(),LogisticRegression()
models = [ebm,lr]

models = train_ensemble_models(df_train,label,models)
df_rocs_cnn_cog_score = calculate_experiment_performance_on_datasets(models,datasets,label)

def get_model_with_higher_auc(df,models):
    return df_rocs.query("set == @set").sort_values('AUC',ascending=False).index[0]

model = ebm
model_col = 'CNN_3Slices_COG_SCORE'
y_pred_proba = model.predict_proba(df_ensemble.drop(['SUBJECT','DATASET','DIAGNOSIS'],axis=1).fillna(0))[:,-1]
df_compare = df_ensemble.copy()
df_compare[model_col] = y_pred_proba

features = df_test.columns[:-1]
show_feature_weights(features,lr.coef_.ravel(),'Logistic Regression - Intercept = %3.2f' % lr.intercept_[0], color = ('r','g'),figsize=(6,6), absolute_values = False,normalized=True)

show(ebm.explain_global())

ebm_local = ebm.explain_local(df_test.drop('DIAGNOSIS',axis=1),df_test['DIAGNOSIS'])
show(ebm_local)

# %%
"""## Experiment 2 using Features Coronal70, Axial8, Sagittal50"""

df_train,df_validation,df_test = get_experiment_sets(df_ensemble,cols_to_drop = ['SUBJECT','DATASET','COGTEST_SCORE'])

datasets=[df_train,df_validation,df_test]
label = 'DIAGNOSIS'
ebm,lr = ExplainableBoostingClassifier(),LogisticRegression()
models = [ebm,lr]

models = train_ensemble_models(df_train,label,models)
df_rocs_cnn3slices = calculate_experiment_performance_on_datasets(models,datasets,label)

model = lr
model_col = 'CNN_3Slices'
y_pred_proba = model.predict_proba(df_ensemble.drop(['SUBJECT','DATASET','DIAGNOSIS','COGTEST_SCORE'],axis=1).fillna(0))[:,-1]
df_compare[model_col] = y_pred_proba

features = df_test.columns[:-1]
show_feature_weights(features,lr.coef_.ravel(),'Logistic Regression - Intercept = %3.2f' % lr.intercept_[0], color = ('r','g'),figsize=(6,6), absolute_values = False,normalized=True)

show(ebm.explain_global())

ebm_local = ebm.explain_local(df_test.drop('DIAGNOSIS',axis=1),df_test['DIAGNOSIS'])
show(ebm_local)

# %%
"""## Experiment 3 using Features Coronal70, Axial8, Sagittal50 + Demographics"""

demographic_cols = ['AGE','MALE',
                    'YEARS_EDUCATION','HISPANIC',
                    'RACE_WHITE', 'RACE_BLACK', 'RACE_ASIAN', 'WIDOWED']
df_cog_final = pd.read_csv(cognitive_data_predictions_final_path).set_index("IMAGE_DATA_ID")
df_ensemble_demographics = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True)

df_train,df_validation,df_test = get_experiment_sets(df_ensemble_demographics,cols_to_drop = ['SUBJECT','DATASET','COGTEST_SCORE'])

datasets=[df_train,df_validation,df_test]
label = 'DIAGNOSIS'
ebm,lr = ExplainableBoostingClassifier(),LogisticRegression()
models = [ebm,lr]

models = train_ensemble_models(df_train,label,models)
df_rocs_cnn3slices_demographics = calculate_experiment_performance_on_datasets(models,datasets,label)

model = ebm
model_col = 'CNN_3Slices_Demographics'

df_predict = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True).fillna(0).sort_index()
y_pred_proba = model.predict_proba(df_predict.drop(['SUBJECT','DATASET','DIAGNOSIS','COGTEST_SCORE'],axis=1))[:,-1]
df_compare[model_col] = y_pred_proba

features = df_test.columns[:-1]
show_feature_weights(features,lr.coef_.ravel(),'Logistic Regression - Intercept = %3.2f' % lr.intercept_[0], color = ('r','g'),figsize=(6,6), absolute_values = False,normalized=True)

show(ebm.explain_global())

ebm_local = ebm.explain_local(df_test.drop('DIAGNOSIS',axis=1),df_test['DIAGNOSIS'])
show(ebm_local)

# %%
"""## Experiment 4 - Features Coronal70, Axial8, Sagittal50 + Demographics + CDRSB"""

demographic_cols = ['AGE','MALE',
                    'YEARS_EDUCATION','HISPANIC','CDRSB',
                    'RACE_WHITE', 'RACE_BLACK', 'RACE_ASIAN', 'WIDOWED']
df_cog_final = pd.read_csv(cognitive_data_predictions_final_path).set_index("IMAGE_DATA_ID")
df_ensemble_demographics = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True)

df_train,df_validation,df_test = get_experiment_sets(df_ensemble_demographics,cols_to_drop = ['SUBJECT','DATASET','COGTEST_SCORE'])

datasets=[df_train,df_validation,df_test]
label = 'DIAGNOSIS'
ebm,lr = ExplainableBoostingClassifier(),LogisticRegression()
models = [ebm,lr]

models = train_ensemble_models(df_train,label,models)
df_rocs_cnn3slices_demographics_cdrsb = calculate_experiment_performance_on_datasets(models,datasets,label)

model = lr
model_col = 'CNN_3Slices_Demographics_CDRSB'

df_predict = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True).fillna(0).sort_index()
y_pred_proba = model.predict_proba(df_predict.drop(['SUBJECT','DATASET','DIAGNOSIS','COGTEST_SCORE'],axis=1))[:,-1]
df_compare[model_col] = y_pred_proba

features = df_test.columns[:-1]
show_feature_weights(features,lr.coef_.ravel(),'Logistic Regression - Intercept = %3.2f' % lr.intercept_[0], color = ('r','g'),figsize=(6,6), absolute_values = False,normalized=True)

show(ebm.explain_global())

ebm_local = ebm.explain_local(df_test.drop('DIAGNOSIS',axis=1),df_test['DIAGNOSIS'])
show(ebm_local)

# %%
"""## Experiment 5 Bonus - Just CDRSB"""

demographic_cols = ['CDRSB']

df_cog_final = pd.read_csv(cognitive_data_predictions_final_path).set_index("IMAGE_DATA_ID")
df_ensemble_demographics = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True)

df_train,df_validation,df_test = get_experiment_sets(df_ensemble_demographics,cols_to_drop = ['DATASET','COGTEST_SCORE'])

datasets=[df_train,df_validation,df_test]
label = 'DIAGNOSIS'
ebm,lr = ExplainableBoostingClassifier(),LogisticRegression()
models=[DummyModel(slice='CDRSB')]

models = train_ensemble_models(df_train,label,models)
df_rocs_cdrsb = calculate_experiment_performance_on_datasets(models,datasets,label)

model = DummyModel(slice='CDRSB')
model_col = 'CDRSB'

df_predict = df_cog_final[demographic_cols].merge(df_ensemble,right_index=True,left_index=True).fillna(0).sort_index()
y_pred_proba = model.predict_proba(df_predict.drop(['DATASET','DIAGNOSIS','COGTEST_SCORE'],axis=1))[:,-1]
df_compare[model_col] = y_pred_proba

df_compare.to_csv('/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_ALL_SCORES_ENSEMBLE_AD_20211128.csv')
# df_compare.to_csv('/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_ALL_SCORES_ENSEMBLE_MCI_20211127.csv')

# %%

"""# Compare All Experiments
  1. CNN Coronal
  2. CNN Axial
  3. CNN Sagittal
  4. Ensemble CNNs
  5. Ensemble CNNs + CogTest Score
  6. Ensemble CNNs + Demographics
  7. Ensemble CNNs + Demographics + CDRSB
  8. CDRSB
"""
datasets = [df_compare.query("DATASET == 'train'"),
            df_compare.query("DATASET == 'validation'"),
            df_compare.query("DATASET == 'test'")]

df_compare = pd.read_csv('/home/lucas/projects/mmml-alzheimer-diagnosis/data/PREDICTIONS_ALL_SCORES_ENSEMBLE_AD_20211128.csv') 
models = ['CNN_SCORE_AXIAL_23','CNN_SCORE_CORONAL_43', 'CNN_SCORE_SAGITTAL_26',
       'CNN_3Slices_COG_SCORE', 'CNN_3Slices', 'CNN_3Slices_Demographics',
       'CNN_3Slices_Demographics_CDRSB', 'CDRSB']
df_rocs = calculate_rocs_on_datasets(models = models,datasets=datasets,label ='DIAGNOSIS')
# compare_ensembles_rocs_on_dataset(df_compare.query("DATASET == 'test'"),'DIAGNOSIS',models)
# %%


models = ['CNN_SCORE_AXIAL_23','CNN_SCORE_CORONAL_43', 'CNN_SCORE_SAGITTAL_26',
       'CNN_3Slices']
df_rocs = calculate_rocs_on_datasets(models = models,datasets=datasets,label ='DIAGNOSIS')
# compare_ensembles_rocs_on_dataset(df_compare.query("DATASET == 'test'"),'DIAGNOSIS',models)

"""# Next Steps

1. Record optimal cutoff points for each case and generate labels. 
2. Calculate other metrics for bar plot comparison
"""
# %%

from sklearn.metrics import roc_auc_score,roc_curve,auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def get_f1(y_true,y_pred_proba,threshold):
    y_pred = [1 if prob >= threshold else 0 for prob in y_pred_proba]
    return f1_score(y_true,y_pred)

def find_threshold_for_optimal_f1(df,model,label='DIAGNOSIS',min_thresh=0,max_thresh=1,num_thresh=100):
    # df = df_compare.query("DATASET == 'validation'")
    y_pred_proba = df[model]
    y_true = df[label]

    thresholds = np.linspace(min_thresh,max_thresh,num_thresh)
    f1_scores = [get_f1(y_true,y_pred_proba,threshold) for threshold in thresholds]
    max_f1 = np.max(f1_scores)
    opt_thresh = thresholds[np.argmax(f1_scores)]
    plt.plot(thresholds,f1_scores,label='Highest F1 Score %.4f at %.4g' % (max_f1,opt_thresh))
    plt.title("Prediction thresholds x F")
    plt.legend(loc="lower center")
    plt.show()
    return max_f1,opt_thresh

# %%
max_f1 = np.max(f1_scores)
opt_thresh = thresholds[np.argmax(f1_scores)]
plt.plot(thresholds,f1_scores,label='Highest F1 Score %.4f at %.4g' % (max_f1,opt_thresh))
plt.title("Prediction thresholds x F")
plt.legend(loc="lower center")
plt.show()
# %%
