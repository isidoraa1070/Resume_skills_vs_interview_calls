import pandas as pd
from utils.save_model_columns import model_columns  # import global dict

def validate_input_columns(model_name: str, df: pd.DataFrame):
    """
    Check for missing or extra columns against saved feature columns.

    """
    columns_info = model_columns.get(model_name)
    if not columns_info:
        return df 

    required_cols = list(columns_info.keys())
    input_cols = list(df.columns)

    missing = [c for c in required_cols if c not in input_cols]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = [c for c in input_cols if c not in required_cols]
    if extra:
        raise ValueError(f"Extra columns not allowed: {extra}")

    return df[required_cols]
