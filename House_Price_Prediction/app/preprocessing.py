#imports
import pandas as pd
from pathlib import Path
import logging
import joblib
import pickle
import csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#preprocessing...
def preprocess(X: pd.DataFrame, models_dir: Path, training_mode: bool) -> pd.DataFrame:
    X_preprocessed = preprocess_continuous_data(X)
    X_preprocessed = preprocess_categorical_data(X_preprocessed, models_dir, training_mode)
    if training_mode:
        create_model(X_preprocessed, models_dir)
    else:
        load_model(X_preprocessed, models_dir)

    return X_preprocessed

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
    X_with_encoded_categorical_features = encode_categorical_features_orchestrator(
        X_preprocessed, models_dir, training_mode)
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
            raise Exception('Loading Error')
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

def get_continuous_columns_in_dataframe(dataframe):
    continuous_columns = dataframe.select_dtypes(include='number').columns
    return continuous_columns

def get_categorical_columns(dataframe: pd.DataFrame) -> [str]:
    categorical_columns_list = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns_list

#model_creation & Loading...
def create_model(X_preprocessed, models_dir):
    train = X_preprocessed.copy()
    y = train['SalePrice']
    X = train.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    test_score = linreg.score(X_test, y_test)
    print("Test_Score:", test_score)
    dump_model(linreg, models_dir)
    return test_score


def dump_model(linreg, models_dir):
    model_filepath = models_dir / 'model_pickle.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(linreg, file)
    return model_filepath

def load_model(X_preprocessed, models_dir):
    model_filepath = models_dir / 'model_pickle.pkl'
    with open(model_filepath, 'rb') as file:
        mp = pickle.load(file)
    predicted = mp.predict(X_preprocessed)
    csv_write(predicted)

def csv_write(predicted):
    header = ['Sale_Price']
    with open('Sale_Price.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(predicted)