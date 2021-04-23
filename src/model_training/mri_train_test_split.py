import numpy as np
import pandas as pd

def train_test_split_by_subject(df,test_size = 0.3,labels = ['AD','CN'],label_column = 'MACRO_GROUP'):
    
    '''
    Splits the dataset on train and test, at patient level (to avoid data leakage).
    This process executes a stratified random split, that is, it maintains the proportion of each class in the sets.
    
    Parameters
    ----------

    df: Reference dataframe containing information about patients.
    
    test_size: test dataset size. Value must be between bigger than 0 and less than 1.

    labels: Label of the classes.
    
    label_column: Name of the column in the dataframe that contains the classes.
    
    only_ids: Flag to return only a list with IDs of the patients. If False, it returns the entire reference dataframes.

    Returns
    ----------
    Tuple with train and test reference datasets: df_train, df_test
    
    '''

    train = []
    test = []
    df_classes = df[df[label_column].isin(labels)]

    # split by subject id
    subjects = df_classes['SUBJECT'].unique()
    np.random.shuffle(subjects)

    for label in labels:
        subjects = df_classes.query(label_column +" == @label")['SUBJECT'].unique()
        np.random.shuffle(subjects)

        test_subjects_quantity = int(np.ceil(test_size * subjects.shape[0]))
        test_subjects = subjects[:test_subjects_quantity]
        train_subjects = subjects[test_subjects_quantity:]
        
        df_train_cl = df_classes.query("SUBJECT in @train_subjects")
        df_test_cl = df_classes.query("SUBJECT in @test_subjects")
        train.append(df_train_cl)
        test.append(df_test_cl)

    df_train = pd.concat(train).sample(frac=1).reset_index(drop=True)
    df_test = pd.concat(test).sample(frac=1).reset_index(drop=True)

    return df_train,df_test