

def train_validation_test_split(df,proportion = [0.5,0.3,0.2],classes = ['AD','CN'],class_column = ['GROUP']):

    train = []
    validation = []
    test = []
    df_classes = df[df[class_column].isin(classes)]

    for cl in classes:
        df_shuffled = df[df[class_column] == cl].sample(frac=1).reset_index(drop=True)

        df_train_cl = df_shuffled.iloc[:int(np.ceil(proportion[0] * df_shuffled.shape[0]))]
        df_validation_cl = df_shuffled.iloc[int(np.ceil(proportion[0] * df_shuffled.shape[0])):int(np.ceil((proportion[0] + proportion[1]) * df_shuffled.shape[0]))]
        df_test_cl = df_shuffled.iloc[int(np.ceil((proportion[0] + proportion[1]) * df_shuffled.shape[0])):]

        train.append(df_train_cl)
        validation.append(df_validation_cl)
        test.append(df_test_cl)

    df_train = pd.concat(train).sample(frac=1).reset_index(drop=True)
    df_validation = pd.concat(validation).sample(frac=1).reset_index(drop=True)
    df_test = pd.concat(test).sample(frac=1).reset_index(drop=True)
    return df_train,df_validation,df_test

def train_test_split(df,proportion = [0.8,0.2],classes = ['AD','CN'],class_column = ['GROUP']):

    train = []
    test = []
    df_classes = df[df[class_column].isin(classes)]
    
    for cl in classes:
        df_shuffled = df_classes[df_classes[class_column] == cl].sample(frac=1).reset_index(drop=True)

        df_train_cl = df_shuffled.iloc[:int(np.ceil(proportion[0] * df_shuffled.shape[0]))]
        df_test_cl = df_shuffled.iloc[int(np.ceil((proportion[0]) * df_shuffled.shape[0])):]

        train.append(df_train_cl)
        test.append(df_test_cl)

    df_train = pd.concat(train).sample(frac=1).reset_index(drop=True)
    df_test = pd.concat(test).sample(frac=1).reset_index(drop=True)
    return df_train,df_test