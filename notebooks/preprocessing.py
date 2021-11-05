#!/usr/bin/env python
# coding: utf-8

# # PREPROCESSING

# In[5]:


import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# In[6]:


def preprocess(X: pd.DataFrame, models_dir: Path, training_mode: bool) -> pd.DataFrame:
    X_preprocessed = preprocess_continuous_data(X)
    X_preprocessed = preprocess_categorical_data(X_preprocessed, models_dir, training_mode)
    return X_preprocessed

### Get continuous columns from data frame
def get_continuous_columns_in_dataframe(dataframe):
    continuous_columns = dataframe.select_dtypes(include='number').columns
    return continuous_columns

## Get categorical columns from data frame
def get_categorical_columns(dataframe: pd.DataFrame) -> [str]:
    categorical_columns_list = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns_list

### Continuous data
def preprocess_continuous_data(X):
    X_preprocessed = X.copy()
    continuous_columns = get_continuous_columns_in_dataframe(X_preprocessed)
    for column_name in continuous_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(0)
    return X_preprocessed

## Categorical data
def preprocess_categorical_data(X: pd.DataFrame, models_dir: Path, training_mode: bool) -> pd.DataFrame:
    X_preprocessed = impute_missing_categorical_data(X)
    X_with_encoded_categorical_features = encode_categorical_features_orchestrator(X_preprocessed, models_dir, training_mode)
    return X_with_encoded_categorical_features

def impute_missing_categorical_data(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = X.copy()
    categorical_columns = get_categorical_columns(X_preprocessed)
    for column_name in categorical_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna('Missing value')
    return X_preprocessed

def encode_categorical_features_orchestrator(X: pd.DataFrame, models_dir: Path,
                                             training_mode: bool = False) -> pd.DataFrame:
    one_hot_encoder = get_categorical_encoder(X, models_dir, training_mode)
    X_with_continuous_data_and_encoded_categorical_data = encode_categorical_features(one_hot_encoder, X)
    return X_with_continuous_data_and_encoded_categorical_data

def get_categorical_encoder(X: pd.DataFrame, models_dir: Path, training_mode: bool):
    encoder_filepath = models_dir / 'encoder_categorical.joblib'
    if training_mode:
        logging.error('Generating a new encoder')
        categorical_columns = get_categorical_columns(X)
        one_hot_encoder = OneHotEncoder(handle_unknown='ignore', dtype=int, sparse=True)
        one_hot_encoder.fit(X[categorical_columns])
        joblib.dump(one_hot_encoder, encoder_filepath)
    else:
        logging.error('Loading an existng encoder')
        if encoder_filepath.is_file():
            one_hot_encoder = joblib.load(encoder_filepath)
        else:
            raise Exception('The blablablm')
    return one_hot_encoder

def encode_categorical_features(one_hot_encoder, X):
    categorical_columns = get_categorical_columns(X)
    continuous_columns = get_continuous_columns_in_dataframe(X)
    encoded_categorical_data_matrix = one_hot_encoder.transform(X[categorical_columns])
    encoded_data_columns = one_hot_encoder.get_feature_names(categorical_columns)
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=X.index)
    X_with_continuous_data_and_encoded_categorical_data = X.copy()[continuous_columns].join(encoded_categorical_data_df)
    return X_with_continuous_data_and_encoded_categorical_data


# In[ ]:




