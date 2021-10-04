# %%
import pandas as pd
import numpy as np

from train_test_split import train_test_split_by_subject

# %%

def execute_ensemble_preparation(ensemble_data_path,test_size=0.41,validation_size=0.28):
    
    '''
    Prepare ensemble data for training process by splitting the data between train, validation and test set.

    The validation and test sets should remain the same for all experiments: MRI, cognitive tests and ensemble.

    Training data can vary from experiment to experiment. 
    For example, MRI CNNs training can contain more images than the images aligned with the ensemble data. 
    The same holds true for cognitive tests data.
    
    Parameters:
    ------------
    ensemble_data_path: path where the preprocessed ensemble data reference is located.

    test_size: size of the test set. Has to be bigger than 0 and less than 1.

    validation_size: size of the validation set. Has to be bigger than 0 and less than 1.

    Results
    ------------

    Saves a prepared ensemble reference file with a DATASET flag (train, test,validation or nan)
    '''

    df_ensemble = pd.read_csv(ensemble_data_path)
    print("Spliting ensemble data in train, validation and test...")
    df_ensemble.sort_values('SUBJECT',inplace=True)
    df_train,df_validation = train_test_split_by_subject(df_ensemble.query("CONFLICT_DIAGNOSIS == False"),test_size = validation_size,labels = [0,1],label_column='DIAGNOSIS',random_seed=42)
    df_train,df_test = train_test_split_by_subject(df_train,test_size = test_size,labels = [0,1],label_column='DIAGNOSIS',random_seed=42)
    print("Ensemble train size:",df_train.shape)
    print("Ensemble validation size:",df_validation.shape)
    print("Ensemble test size:",df_test.shape)
    
    df_ensemble['DATASET'] = np.nan
    df_train['DATASET'] = 'train'
    df_validation['DATASET'] = 'validation'
    df_test['DATASET'] = 'test'

    df_ensemble_processed = pd.concat([df_train,df_validation,df_test,df_ensemble.query("CONFLICT_DIAGNOSIS == True")])
    df_ensemble_processed.to_csv(ensemble_data_path.replace('PREPROCESSED','PROCESSED'),index=False)
    
    return df_train,df_validation,df_test

# %%
if __name__ == '__main__':
    ensemble_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
    df_train,df_validation,df_test = execute_ensemble_preparation(ensemble_data_path,test_size=0.33,validation_size=0.25)
