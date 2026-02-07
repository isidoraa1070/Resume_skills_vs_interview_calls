from sklearn.linear_model import LinearRegression

models = {}

def create_model(
    model_name: str,
    fit_intercept: bool = True,
    positive: bool = False
):
    
    """

    Creates and stores models in the global dict, using fit_intercept and positive as config options.

    """
    if model_name in models:
        raise ValueError(f"Model '{model_name}' already exists.")

    model = LinearRegression(
        fit_intercept=fit_intercept,
        positive=positive
    )

    models[model_name] = model

