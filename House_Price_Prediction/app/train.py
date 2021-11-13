import logging
from pathlib import Path
import pandas as pd
import joblib
from app.preprocessing import preprocess
from app.preprocessing import create_model

def train(train_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(train_filepath)

    X_Preprocess = preprocess(df, MODELS_DIR, training_mode)
    dictionary = X_Preprocess.to_dict();

    return "Model Dir:", MODELS_DIR





