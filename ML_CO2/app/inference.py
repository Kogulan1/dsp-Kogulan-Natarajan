import pandas as pd
from models.app import preprocess

def inference(test_filepath, MODELS_DIR, training_mode):
    df = pd.read_csv(test_filepath)

    X_Preprocess = preprocess(df, MODELS_DIR, training_mode)
    return "`Co2 Emission Predicted and saved as Co2_Emission.csv "

