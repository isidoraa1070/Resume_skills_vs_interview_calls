import pandas as pd
import os
import pickle
from app.create_model import models
from utils.validate_input_columns import validate_input_columns


def predict(model_name: str, data: list):
    """

    Checks for model existence, loads it if in memory, 
    validates input data against saved feature columns, and returns predictions.

    """

    if model_name not in models:
        path = f"models/{model_name}.pkl"
        if not os.path.exists(path):
            raise ValueError(f"Model '{model_name}' does not exist.")
        with open(path, "rb") as f:
            models[model_name] = pickle.load(f)

    model = models[model_name]

    df = pd.DataFrame(data)

    validated_df = validate_input_columns(model_name, df)

    preds = model.predict(validated_df)

    return preds.tolist()

