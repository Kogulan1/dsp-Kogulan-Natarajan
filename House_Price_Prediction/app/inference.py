import logging
from pathlib import Path
import pandas as pd
import joblib
from app.preprocessing import preprocess

def inference(test_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(test_filepath)

    X_Preprocess = preprocess(df, MODELS_DIR, training_mode)
    return "`Sales Price Predicted and saved as Sale_Price.csv "