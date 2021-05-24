import numpy as np
import pandas as pd

def train_test_split_by_subject(df,test_size = 0.3,labels = ['AD','CN'],label_column='MACRO_GROUP',random_seed=42):
    
    '''
    Splits the dataset on train and test, at patient level (to avoid data leakage).
    
    This process executes a stratified random split, that is, it maintains the proportion of each class in the sets.
    
    Parameters
    ----------

    df: Reference dataframe containing information about patients.
    
    test_size: test dataset size. Value must be between bigger than 0 and less than 1.

    labels: Label of the classes.
    
    label_column: Column containing the label class to filter the final train and test set.


    Returns
    ----------
    Tuple with train and test reference datasets: df_train, df_test
    
    '''

    train = []
    test = []
    df_classes = df[df[label_column].isin(labels)]
    rng = np.random.default_rng(random_seed)
    patients_by_class = []

    for label in labels:
      label_patients = df_classes.query(label_column + "== @label")['SUBJECT'].unique()
      patients_by_class.append(label_patients)

    if len(patients_by_class) == 3:
      patients_all_classes = list(set(patients_by_class[0]) & set(patients_by_class[1]) & set(patients_by_class[2]))
      patients_separated_all_classes =  np.array_split(patients_all_classes,3)
    else:
      patients_all_classes = list(set(patients_by_class[0]) & set(patients_by_class[1]))
      patients_separated_all_classes =  np.array_split(patients_all_classes,2)

    for ii,label in enumerate(labels):
        patients_from_other_fold_classes = list(set(patients_all_classes) - set(patients_separated_all_classes[ii]))
        subjects = df_classes.query(label_column +" == @label and SUBJECT not in @patients_from_other_fold_classes")['SUBJECT'].unique()
        rng.shuffle(subjects)
        
        test_subjects_quantity = int(np.ceil(test_size * subjects.shape[0]))
        test_subjects = subjects[:test_subjects_quantity]
        train_subjects = subjects[test_subjects_quantity:]
        
        df_train_cl = df_classes.query("SUBJECT in @train_subjects")
        df_test_cl = df_classes.query("SUBJECT in @test_subjects")
        train.append(df_train_cl)
        test.append(df_test_cl)

    df_train = pd.concat(train).sample(frac=1).reset_index(drop=True).query(label_column + " in @labels")
    df_test = pd.concat(test).sample(frac=1).reset_index(drop=True).query(label_column + " in @labels")

    return df_train,df_test
