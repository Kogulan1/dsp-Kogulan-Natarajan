#imports
import pandas as pd
import logging
import joblib
import pickle
import csv

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from models.app import feature_selection_continuous_columns


#preprocessing...
def preprocess(X, models_dir, training_mode):
    X_preprocessed = preprocess_continuous_data(X)
    #X_preprocessed = preprocess_categorical_data(X_preprocessed, models_dir, training_mode)

    if training_mode:
       X_Selected = feature_selection_continuous_columns(X_preprocessed, models_dir)
       print (X_Selected)
       create_model(X_Selected, models_dir)
    else:
        feature_list = ['Engine Size(L)', 'Cylinders','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (L/100 km)','Fuel Consumption Comb (mpg)']
        X_Selected = X_preprocessed[feature_list]


        load_model(X_Selected, models_dir)



    return X_Selected

### Continuous data
def preprocess_continuous_data(X):
    X_preprocessed = X.copy()
    continuous_columns = get_continuous_columns_in_dataframe(X_preprocessed)
    X_preprocessed = X_preprocessed[continuous_columns]
    for column_name in continuous_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna(0)
    return X_preprocessed

## Categorical data
def preprocess_categorical_data(X, models_dir, training_mode):
    X_preprocessed = impute_missing_categorical_data(X)
    X_with_encoded_categorical_features = encode_categorical_features_orchestrator(
        X_preprocessed, models_dir, training_mode)
    return X_with_encoded_categorical_features

def impute_missing_categorical_data(X):
    X_preprocessed = X.copy()
    categorical_columns = get_categorical_columns(X_preprocessed)
    for column_name in categorical_columns:
        X_preprocessed[column_name] = X_preprocessed[column_name].fillna('Missing value')
    return X_preprocessed

def encode_categorical_features_orchestrator(X, models_dir,
                                             training_mode: bool = False):
    one_hot_encoder = get_categorical_encoder(X, models_dir, training_mode)
    X_with_continuous_data_and_encoded_categorical_data = encode_categorical_features(one_hot_encoder, X)
    return X_with_continuous_data_and_encoded_categorical_data

def get_categorical_encoder(X, models_dir, training_mode):
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

def get_categorical_columns(dataframe) -> [str]:
    categorical_columns_list = list(dataframe.select_dtypes(include='object').columns)
    return categorical_columns_list


def create_model(X_preprocessed, models_dir):
    train = X_preprocessed.copy()
    y = train['CO2 Emissions(g/km)']
    X = train.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    test_score = linear_regression.score(X_test, y_test)
    print("Score:", test_score)
    dump_model(linear_regression, models_dir)
    return test_score

#Write model as pickle file
def dump_model(linreg, models_dir):
    model_filepath = models_dir / 'model_pickle.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(linreg, file)
    print("Model Stored as model_pickle in models folder")
    return model_filepath

#Load model for Inference
def load_model(X_preprocessed, models_dir):
    model_filepath = models_dir / 'model_pickle.pkl'
    with open(model_filepath, 'rb') as file:
        mp = pickle.load(file)
    predicted = mp.predict(X_preprocessed)
    csv_write(predicted)


#Write Predicted values as CSV
def csv_write(predicted):
    header = ['CO2_Emissions']
    with open('CO2_Emissions.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(predicted)



def flatten_list(feature_selected):
    flat_list = []
    # Iterate through the outer list
    for element in feature_selected:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list