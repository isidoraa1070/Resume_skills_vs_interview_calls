import pandas as pd

def prepare_dataset(
    csv_path: str,
    target_column: str = "interview_calls",
    drop_columns: list | None = None,
    binary_columns: list | None = None,
    encode_degree: bool = False,
    save_path: str | None = None
):
    """
    Loads dataset, cleans and encodes features,
    and returns features (X) and target (y).
    
    Parameters:
    - csv_path: path to raw CSV
    - target_column: name of target variable
    - drop_columns: list of columns to drop
    - binary_columns: list of Yes/No columns to convert
    - encode_degree: whether to one-hot encode degree column
    - save_path: optional path to save cleaned dataset

    """

    # 1. Load data
    data = pd.read_csv(csv_path)

    # 2. Drop unwanted columns
    if drop_columns:
        existing = [c for c in drop_columns if c in data.columns]
        if existing:
            data = data.drop(columns=existing)

    # 3. Convert Yes/No columns to numeric
    if binary_columns:
        binary_map = {"Yes": 1, "No": 0}
        for col in binary_columns:
            if col in data.columns:
                data[col] = data[col].map(binary_map)

    # 4. One-hot encode degree column if requested
    if encode_degree and "degree" in data.columns:
        data = pd.get_dummies(data, columns=["degree"], drop_first=True)

    # 5. Define target and features
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    y = data[target_column]
    X = data.drop(columns=[target_column])

    # 6. Save cleaned dataset if path provided
    if save_path:
        data.to_excel(save_path, index=False)

    return X, y
