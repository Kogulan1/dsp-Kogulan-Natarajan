import pandas as pd
from app.preprocessing import preprocess
import pickle

def inference(test_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(test_filepath)
    X_Preprocess = preprocess(df, MODELS_DIR, training_mode)
    X_predicted = load_model(X_Preprocess,MODELS_DIR)
    return X_predicted

def load_model(X_preprocessed, models_dir):
    model_filepath = models_dir / 'predictor.pkl'
    with open(model_filepath, 'rb') as file:
        mp = pickle.load(file)
    predicted = mp.predict(X_preprocessed)
    return predicted