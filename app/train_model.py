from app.create_model import models
from utils.prepare_dataset import prepare_dataset
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os


def train_model(
    model_name: str,
    csv_path: str,
    target_column="interview_calls",
    drop_columns=None,
    binary_columns=None,
    encode_degree=False
    ):
    """
    Trains the specified model using the provided dataset and returns performance metrics
    
    """

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' does not exist.")

    model = models[model_name]

    X, y = prepare_dataset(
        csv_path,
        target_column=target_column,
        drop_columns=drop_columns,
        binary_columns=binary_columns,
        encode_degree=encode_degree
    )

    model.fit(X, y)
    preds = model.predict(X)

    metrics = {
        "mse": mean_squared_error(y, preds),
        "r2": r2_score(y, preds),
        "samples": len(X)
    }

    os.makedirs("models", exist_ok=True)
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

    return metrics, X
