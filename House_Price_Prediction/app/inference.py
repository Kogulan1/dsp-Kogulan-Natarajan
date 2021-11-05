import logging
from pathlib import Path
import pandas as pd
import joblib
from app.preprocessing import preprocess

def inference(test_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(test_filepath)
    mode = training_mode
    models_dir = MODELS_DIR
    X_Preprocess = preprocess(df, models_dir, mode)
    return X_Preprocess