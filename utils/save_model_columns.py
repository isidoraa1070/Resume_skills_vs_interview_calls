model_columns = {}  
def save_model_columns(model_name: str, X):
    """
    Saves the feature columns and their data types for a given model after training.

    """
    # dict: {"col_name": "dtype"}
    columns_info = {col: str(dtype) for col, dtype in X.dtypes.items()}
    model_columns[model_name] = columns_info