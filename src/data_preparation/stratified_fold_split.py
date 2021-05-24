import numpy as np
import pandas as pd

def stratified_fold_split_by_subject(df, n_splits=10, labels = ['AD','CN'], label_column = 'MACRO_GROUP', random_seed=42, return_indices=False):

    '''
    Provides train/test fold indices to split data at patient level, in order to avoid data leakage.
    
    This process executes a stratified random split, that is, it maintains the proportion of each class in the sets.
    
    Parameters
    ----------

    df: Reference dataframe containing information about patients.
    
    n_splits: number to determine the amount of fold splits in data.

    labels: Label of the classes.
    
    label_column: Column containing the label class to filter the final train and test set.

    return_indices: Flag to return the train/test indices. If False, it returns the entire reference dataframes.

    Returns
    ----------
    Tuple with train and test reference datasets: df_train, df_test

    Example
    ----------

    y = df_adni_merge['DIAGNOSIS']
    X = df_adni_merge.drop(['DIAGNOSIS'],axis=1)

    for train_index, test_index in stratified_fold_split_by_subject(df, n_splits=10,labels = ['AD','CN'],label_column = 'DIAGNOSIS_BASELINE',return_indices=True):
        X_train = df_adni_merge.query(index in @)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ...
        ...
        ...

    ---------

    y = df_adni_merge['DIAGNOSIS']
    X = df_adni_merge.drop(['DIAGNOSIS'],axis=1)

    results = sklearn.model_selection.cross_validate(
                                ExplainableBoostingClassifier(**ebm_params),
                                X,y,
                                cv=stratified_fold_split_by_subject(df, n_splits=10,labels = ['AD','CN'],label_column = 'DIAGNOSIS_BASELINE',return_indices=True),
                                scoring=my_auc,n_jobs=-1)
    '''

    train = []
    test = []
    df_classes = df[df[label_column].isin(labels)].copy()
    df_classes['FOLD'] = 0

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

        n_subjects = subjects.shape[0]
        fold_size = int(np.ceil(n_subjects / n_splits))

        subjects_by_fold = np.array_split(subjects,n_splits)
        for split in range(n_splits):
            fold_subjects = subjects_by_fold[split]
            df_classes.loc[df_classes['SUBJECT'].isin(fold_subjects),'FOLD'] = split

    if return_indices:
        for split in range(n_splits):
            train_index = df_classes.query("FOLD != @split").index
            test_index = df_classes.query("FOLD == @split").index
            yield train_index,test_index
    else:
        return df_classes

class StratifiedSubjectKFold:
    def __init__(self,df,
                    n_splits=10, 
                    labels = [0,1], 
                    label_column = 'MACRO_GROUP', 
                    random_seed=42, 
                    return_indices=True):
        self.df = df.copy()
        self.n_splits = n_splits
        self.labels = labels
        self.label_column = label_column
        self.random_seed = random_seed
        self.return_indices = return_indices

    '''
    Provides train/test fold indices to split data at patient level, in order to avoid data leakage.
    
    This process executes a stratified random split, that is, it maintains the proportion of each class in the sets.
    
    Parameters
    ----------

    df: Reference dataframe containing information about patients.
    
    n_splits: number to determine the amount of fold splits in data.

    labels: Label of the classes.
    
    label_column: Column containing the label class to filter the final train and test set.

    return_indices: Flag to return the train/test indices. If False, it returns the entire reference dataframes.

    Returns
    ----------
    Tuple with train and test reference datasets: df_train, df_test

    Example
    ----------

    y = df_adni_merge['DIAGNOSIS']
    X = df_adni_merge.drop(['DIAGNOSIS'],axis=1)

    for train_index, test_index in StratifiedSubjectKFold(df, n_splits=10,labels = ['AD','CN'],label_column = 'DIAGNOSIS_BASELINE',return_indices=True).split(X,y):
        X_train = df_adni_merge.query(index in @)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ...
        ...
        ...

    ---------

    y = df_adni_merge['DIAGNOSIS']
    X = df_adni_merge.drop(['DIAGNOSIS'],axis=1)

    results = sklearn.model_selection.cross_validate(
                                ExplainableBoostingClassifier(**ebm_params),
                                X,y,
                                cv=StratifiedSubjectKFold(df, n_splits=10,labels = ['AD','CN'],label_column = 'DIAGNOSIS_BASELINE',return_indices=True),
                                scoring=my_auc,n_jobs=-1)
    '''

    def split(self, X, y, groups=None):
        
        train = []
        test = []
        df_classes = self.df[self.df[self.label_column].isin(self.labels)].copy()
        df_classes['FOLD'] = 0

        rng = np.random.default_rng(self.random_seed)

        patients_by_class = []

        for label in self.labels:
            label_patients = df_classes.query(self.label_column + "== @label")['SUBJECT'].unique()
            patients_by_class.append(label_patients)

        if len(patients_by_class) == 3:
            patients_all_classes = list(set(patients_by_class[0]) & set(patients_by_class[1]) & set(patients_by_class[2]))
            patients_separated_all_classes =  np.array_split(patients_all_classes,3)
        else:
            patients_all_classes = list(set(patients_by_class[0]) & set(patients_by_class[1]))
            patients_separated_all_classes =  np.array_split(patients_all_classes,2)

        for ii,label in enumerate(self.labels):
            
            patients_from_other_fold_classes = list(set(patients_all_classes) - set(patients_separated_all_classes[ii]))
            subjects = df_classes.query(self.label_column +" == @label and SUBJECT not in @patients_from_other_fold_classes")['SUBJECT'].unique()
            rng.shuffle(subjects)

            n_subjects = subjects.shape[0]
            fold_size = int(np.ceil(n_subjects / n_splits))

            subjects_by_fold = np.array_split(subjects,n_splits)
            for split in range(n_splits):
                fold_subjects = subjects_by_fold[split]
                df_classes.loc[df_classes['SUBJECT'].isin(fold_subjects),'FOLD'] = split

        if self.return_indices:
            for split in range(n_splits):
                train_index = df_classes.query("FOLD != @split").index
                test_index = df_classes.query("FOLD == @split").index
                yield train_index,test_index
        else:
            return df_classes



    def get_n_splits(self, X, y, groups=None):
        return self.n_splits