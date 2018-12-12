import pickle
import numpy as np
import pandas as pd

def pkl_this(filename, df):
    '''Saves df as filename. Must include .pkl extension'''
    with open(filename, 'wb') as picklefile:
        pickle.dump(df, picklefile)

def open_pkl(filename):
    '''Must include .pkl extension. Returns object that can be saved to a df.
    '''
    with open(filename,'rb') as picklefile:
        return pickle.load(picklefile)

def log_this(s):
    with open("log.txt", "a") as f:
        f.write("\n" + s + "\n")

def find_str_in_col(s, col, df):
    '''Finds string 's' in column 'col' of dataframe 'df'
    Input string in case format (lower, upper) that matches dataframe
    '''
    return df[df[col].str.contains(s)]
    # df[df['molecule_name'].str.contains('gilteritinib'.upper())]
    # return res


def clean_model_df(df):
    df.columns = [x.lower().strip().replace('_extracted', '') for x in df.columns]
    # features
    df['molecule_name_x'].fillna('none given', inplace=True)
    df['ro3_pass'] = df['ro3_pass'].apply(lambda x: 0 if x == 'N' else 1)
    df['num_lipinski_ro5_violations'].fillna('0', inplace=True)
    # targets
    df['patent'] = df['num_patents'].notnull().astype(int)
    df['approved'] = df['phase'].apply(lambda x: 1 if x == 3 else 1 if x ==4 else 0)
    df.drop(columns=['unnamed: 0_x', 'unnamed: 0_y', 'molecule_name_y', 'molecule_name'] #,'num_patents', 'phase', ]
        , inplace=True)
    df.rename(columns={'molecule_name_x':'molecule_name','classes2':'target_classes'}, inplace=True)
    return df


def percent_null(col, df):
    return len(df[df[col].isnull()]) / len(df) * 100

def to_num(col, df):
    df[col] = pd.to_numeric(df[col])
    return df
