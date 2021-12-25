import pandas as pd
import logging
import joblib



def preprocess(X, models_dir, training_mode):
    X_preprocessed = data_cleaning_orchestrator(X)
    numeric_columns = get_numerical_columns(X_preprocessed)
    categorical_columns = get_categorical_columns(X_preprocessed)
    features = feature_selection(numeric_columns,categorical_columns)
    X_preprocessed = X_preprocessed[features]
    return X_preprocessed

def data_cleaning_orchestrator(X):
    X_preprocessed = handle_inconsist_data(X)
    X_preprocessed = column_rename(X_preprocessed)
    X_preprocessed  = preprocess_missing_values(X_preprocessed)

    return X_preprocessed


def column_rename(X):
    X.columns = X.columns.str.replace(r"\(.*?\)", "")
    X.columns = X.columns.str.rstrip(' ')
    X.columns = X.columns.str.replace(' ', '_')
    return X

def preprocess_missing_values(X):

    X_preprocessed_missing_values = X.apply(lambda x: x.fillna(x.value_counts().index[0]))
    return X_preprocessed_missing_values

def handle_inconsist_data(X):
    X_preprocessed = X.drop(['Fuel Consumption Comb (mpg)'], axis=1)
    return X_preprocessed


def get_numerical_columns(X):
    numeric_columns =  X.select_dtypes(include=['number']).columns.tolist()
    return numeric_columns

def get_categorical_columns(X):
    categorical_columns =  X.select_dtypes(include=['object']).columns.tolist()
    return categorical_columns


def feature_selection(numeric_columns,categorical_columns):
    exclude_feature = ['Make', 'Model', 'Vehicle_Class']
    categorical_features = [col for col in categorical_columns if col not in exclude_feature]
    exclude_feature = exclude_feature + categorical_features
    features = [col for col in numeric_columns if col not in exclude_feature]

    return features





