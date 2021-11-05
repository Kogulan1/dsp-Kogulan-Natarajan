import logging
from pathlib import Path
import pandas as pd
import joblib
from app.preprocessing import preprocess
from app.preprocessing import create_model

def train(train_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(train_filepath)
    mode = training_mode
    models_dir = MODELS_DIR
    X_Preprocess = preprocess(df, models_dir, mode)
    dictionary = X_Preprocess.to_dict();
    return "Model Dir:", models_dir





