import numpy as np
import pandas as pd
# import torch

def train_test_split(df,train_size = 0.8,classes = ['AD','CN'],class_column = ['GROUP'],only_ids=False):
    '''

    Splits the dataset on train and test, at patient level (to avoid data leakage).
    This process executes a stratified random split, that is, it maintains the proportion of each class in the sets.
    

    Parameters
    ---------------

    df: Reference dataframe containing information about patients.
    
    train_size: train dataset size. Value must be between bigger than 0 and less than 1.

    classes: Label of the classes.
    
    class_column: Name of the column in the dataframe that contains the classes.
    
    only_ids: Flag to return only a list with IDs of the patients. If False, it returns the entire reference dataframes.

    '''

    train = []
    test = []
    df_classes = df[df[class_column].isin(classes)]
    
    for cl in classes:
        df_shuffled = df_classes[df_classes[class_column] == cl].sample(frac=1).reset_index(drop=True)
        total_training_samples = int(np.ceil(train_size * df_shuffled.shape[0]))
        df_train_cl = df_shuffled.iloc[:total_training_samples]
        df_test_cl = df_shuffled.iloc[total_training_samples:]

        train.append(df_train_cl)
        test.append(df_test_cl)

    df_train = pd.concat(train).sample(frac=1).reset_index(drop=True)
    df_test = pd.concat(test).sample(frac=1).reset_index(drop=True)

    if only_ids:
        return df_train['SUBJECT'],df_test['SUBJECT']

    return df_train,df_test