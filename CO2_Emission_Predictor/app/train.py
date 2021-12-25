import pandas as pd
from app.preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train(train_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(train_filepath)
    X_Preprocess = preprocess(df, MODELS_DIR, training_mode)

    score,linear_regression = model_fit(X_Preprocess)
    _ = create_model(linear_regression, MODELS_DIR)


    return score,X_Preprocess,X_Preprocess.columns

def model_fit(X_Preprocess):

    X = X_Preprocess.drop(['CO2_Emissions'], axis=1)
    y = X_Preprocess['CO2_Emissions']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    test_score = linear_regression.score(X_test, y_test)
    return test_score,linear_regression

def create_model(linear_regression, models_dir):
    model_filepath = models_dir / 'predictor.pkl'
    with open(model_filepath, 'wb') as file:
        pickle.dump(linear_regression, file)


