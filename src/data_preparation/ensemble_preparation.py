import pandas as pd
import numpy as np

from train_test_split import train_test_split_by_subject

def execute_ensemble_preparation(ensemble_data_path,output_data_path,classes=[0,1,2],test_size=0.25,validation_size=0.25):
    
    '''
    Prepare ensemble data for training process by splitting the data between train, validation and test set.

    The validation and test sets should remain the same for all experiments: MRI, cognitive tests and ensemble.

    Training data can vary from experiment to experiment. 
    For example, MRI CNNs training can contain more images than the images aligned with the ensemble data. 
    The same holds true for cognitive tests data.
    
    Parameters:
    ------------
    ensemble_data_path: path where the preprocessed ensemble data reference is located.

    classes: Classes to filter out final ensemble reference file.
    
    test_size: size of the test set. Has to be bigger than 0 and less than 1.

    validation_size: size of the validation set. Has to be bigger than 0 and less than 1.

    Results
    ------------

    Saves a prepared ensemble reference file with a DATASET flag (train, test,validation or nan)
    '''

    df_ensemble = pd.read_csv(ensemble_data_path)
    print("Spliting ensemble data in train, validation and test...")
    df_ensemble.sort_values('SUBJECT',inplace=True)
    df_no_conflict = df_ensemble.query("CONFLICT_DIAGNOSIS == False")
    df_train,df_validation = train_test_split_by_subject(df_no_conflict,test_size = validation_size,labels = classes,label_column='DIAGNOSIS',random_seed=151)
    corrected_test_size = test_size * (df_no_conflict.shape[0]/df_train.shape[0])
    df_train,df_test = train_test_split_by_subject(df_train,test_size = corrected_test_size,labels = classes,label_column='DIAGNOSIS',random_seed=151)
    print("Ensemble train size:",df_train.shape)
    print("Ensemble validation size:",df_validation.shape)
    print("Ensemble test size:",df_test.shape)

    df_ensemble['DATASET'] = np.nan
    df_train['DATASET'] = 'train'
    df_validation['DATASET'] = 'validation'
    df_test['DATASET'] = 'test'
    df_ensemble_processed = pd.concat([df_train,df_validation,df_test])
    df_ensemble_processed['IMAGE_DATA_ID'] = 'I' + df_ensemble_processed['IMAGEUID'].astype(str)
    print("Data in AD vs CN: \n",df_ensemble_processed.query("MACRO_GROUP in (0,1)")['DATASET'].value_counts())
    print("Data in MCI vs CN: \n",df_ensemble_processed.query("MACRO_GROUP in (0,2)")['DATASET'].value_counts())
    df_ensemble_processed.to_csv(output_data_path)
    return df_ensemble_processed


# if __name__ == '__main__':
#     ensemble_data_path = '/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv'
#     df_train,df_validation,df_test = execute_ensemble_preparation(ensemble_data_path,test_size=0.25,validation_size=0.25)

