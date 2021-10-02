import pandas as pd
import numpy as np
import argparse

def execute_cognitive_data_preprocessing(input_path,output_path,exclude_ecog_tests=True):
    
    '''
    
    Prepare the cognitive tests data for classification based on the ADNIMERGE.csv file.

    Main steps are:
    
    - Normalizing classes to CN, MCI and AD.
    - Selecting only relevant data related to patient demographics and cognitive tests.
    - Encode Race, Gender and Marital Status features.
    - Encode Classes as: CN = 0, AD=1 and MCI=2.
    - Exclude Everyday Cognition Tests (optional).

    Results saved to: <output_path>/COGNITIVE_DATA_PREPROCESSED.csv

    '''
    print("Reading ADNIMERGE.csv file.")
    df_adni_merge = pd.read_csv(input_path+'ADNIMERGE.csv',low_memory=False)
    
    # print("Dropping baseline columns.")
    # baseline_cols = [x for x in df_adni_merge.columns if '_bl' in x and 'DX' not in x] + ['update_stamp']
    # df_adni_merge.drop(baseline_cols,axis=1,inplace=True)

    print("Normalizing classes.")
    df_adni_merge = normalize_classes(df_adni_merge)

    print("Selecting Cognitive Tests and Demographics data.")
    df_adni_merge = select_cognitive_data(df_adni_merge)

    df_adni_merge = df_adni_merge[~df_adni_merge['DX'].isnull()]

    df_adni_merge.rename(inplace=True,
        columns={
        'PTRACCAT':'RACE',
        'PTMARRY':'MARRIED',
        'PTEDUCAT':'YEARS_EDUCATION',
        'PTGENDER':'MALE',
        'PTETHCAT':'HISPANIC',
        'DX':'DIAGNOSIS',
        'DX_bl':'DIAGNOSIS_BASELINE',
        'PTID':'SUBJECT'
    })

    print("Encoding variables.")
    df_adni_merge = encode_variables(df_adni_merge)

    if exclude_ecog_tests:
        print("Excluding Ecog tests.")
        df_adni_merge = exclude_ecog(df_adni_merge)

    print("Saving final data.")
    df_adni_merge.to_csv(output_path + "COGNITIVE_DATA_PREPROCESSED.csv",index=False)

def normalize_classes(df):

    "Normalize diagnosis to CN,AD and MCI."

    df.loc[df['DX'] == 'Dementia','DX'] = 'AD'
    df.loc[df['DX_bl'] == 'LMCI','DX_bl'] = 'MCI'
    df.loc[df['DX_bl'] == 'EMCI','DX_bl'] = 'MCI'
    df.loc[df['DX_bl'] == 'SMC','DX_bl'] = 'CN'
    return df
    
def select_cognitive_data(df):

    neuropsychological_cols = ['CDRSB','ADAS11', 'ADAS13', 'ADASQ4', 'MMSE', 'RAVLT_immediate',
    'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting',
    'LDELTOTAL', 'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem',
    'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
    'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang',
    'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt','EcogSPTotal']

    demographics_cols = ['AGE','PTGENDER', 'PTEDUCAT','PTETHCAT', 'PTRACCAT', 'PTMARRY']

    id_cols = ['RID', 'PTID', 'VISCODE', 'SITE', 'COLPROT', 'ORIGPROT', 'EXAMDATE','IMAGEUID','DX','DX_bl']

    return df[id_cols + demographics_cols + neuropsychological_cols]

def encode_variables(df):
    
    "Encode Race, Gender, Marital Status and Diagnosis variables. CN = 0, AD = 1 and MCI = 2."

    df.loc[df['RACE'].isin(["More than one",'Unkown','Unknown','Hawaiian/Other PI','Am Indian/Alaskan']),'RACE'] = 'Other races'
    df['RACE_WHITE'] = (df['RACE'] == 'White').astype(int)
    df['RACE_BLACK'] = (df['RACE'] == 'Black').astype(int)
    df['RACE_ASIAN'] = (df['RACE'] == 'Asian').astype(int)

    df.loc[df['HISPANIC'] == 'Not Hisp/Latino','HISPANIC'] = 0
    df.loc[df['HISPANIC'] == 'Unknown','HISPANIC'] = 0
    df.loc[df['HISPANIC'] == 'Hisp/Latino','HISPANIC'] = 1

    df['IMAGEUID'] = df['IMAGEUID'].fillna(999999)
    df['IMAGEUID'] = df['IMAGEUID'].astype(int)

    df.loc[df['DIAGNOSIS'] == 'AD','DIAGNOSIS'] = 1
    df.loc[df['DIAGNOSIS'] == 'CN','DIAGNOSIS'] = 0
    df.loc[df['DIAGNOSIS'] == 'MCI','DIAGNOSIS'] = 2

    df.loc[df['MALE'] == 'Male','MALE'] = 1
    df.loc[df['MALE'] == 'Female','MALE'] = 0

    df['WIDOWED'] = (df['MARRIED'] == 'Widowed').astype(int)
    df['DIVORCED'] = (df['MARRIED'] == 'Divorced').astype(int)
    df['NEVER_MARRIED'] = (df['MARRIED'] == 'Never married').astype(int)
    df['MARRIED'] = (df['MARRIED'] == 'Married').astype(int)
    return df

def exclude_ecog(df):
    ecog_cols = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan',
    'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang',
    'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt',
    'EcogSPTotal','LDELTOTAL','DIGITSCOR']
    df = df.drop(ecog_cols,axis=1)
    return df

arg_parser = argparse.ArgumentParser(description='Executes Data Preparation for Cognitive test data.')

arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=False,
                    default='/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/',
                    help='Input directory of cognitive data')

arg_parser.add_argument('-o','--output',
                    metavar='output',
                    type=str,
                    required=False,
                    default='/content/gdrive/MyDrive/Lucas_Thimoteo/data/tabular/',
                    help='Output directory of cognitive data')

arg_parser.add_argument('-e','--exclude',
                    metavar='exclude_ecog_tests',
                    type=bool,
                    required=False,
                    default=True,
                    help='Flag to exclude the everyday cognition tests.')

args = arg_parser.parse_args()

if __name__ == '__main__':
    execute_cognitive_data_preprocessing(args.input,args.output,args.exclude)
