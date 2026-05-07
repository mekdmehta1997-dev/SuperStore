import joblib
import pandas as pd

from src.prediction_logger import log_prediction

MODEL_PATH = "models/model.pkl"

# ==================================================
# LOAD MODEL
# ==================================================
model = joblib.load(MODEL_PATH)

# ==================================================
# PREDICTION FUNCTION
# ==================================================
def predict_discount(data):

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    # SAVE LOG
    log_prediction(data, prediction)

    return float(prediction)