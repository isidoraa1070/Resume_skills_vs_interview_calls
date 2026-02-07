from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from app.create_model import create_model, models
from utils.save_model_columns import save_model_columns
from app.train_model import train_model
from app.predict_model import predict
import pandas as pd
from utils.save_model_columns import model_columns  # global dict to store feature columns for each model

app = FastAPI(title="Linear Regression Service")

class CreateModelRequest(BaseModel):
    model_name: str
    model_type: Optional[str] = "linear"
    fit_intercept: Optional[bool] = True
    positive: Optional[bool] = False


class TrainRequest(BaseModel):
    model_name: str
    csv_path: str
    target_column: Optional[str] = "interview_calls"
    drop_columns: Optional[List[str]] = None
    binary_columns: Optional[List[str]] = None
    encode_degree: Optional[bool] = False

class PredictRequest(BaseModel):
    model_name: str
    data: List[Dict]

@app.post("/create_model")
def create_model_api(req: CreateModelRequest):
    if req.model_type != "linear":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type '{req.model_type}'."
        )
    
    if req.model_name in models:
        # treba HTTP 400
        raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' already exists.")

    try:
        create_model(
            model_name=req.model_name,
            fit_intercept=req.fit_intercept,
            positive=req.positive
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": f"Model '{req.model_name}' created successfully.",
        "config": {
            "fit_intercept": req.fit_intercept,
            "positive": req.positive
        }
    }


@app.post("/train_model")
def train_model_api(req: TrainRequest):
    if req.model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{req.model_name}' doesn't exist. Create it first.")

    try:
        # Trenira model i dobija metrike + X
        metrics, X = train_model(
            model_name=req.model_name,
            csv_path=req.csv_path,
            target_column=req.target_column,
            drop_columns=req.drop_columns,
            binary_columns=req.binary_columns,
            encode_degree=req.encode_degree
        )

        # Sačuvaj feature kolone za predict
        save_model_columns(req.model_name, X)

        return {
            "message": f"Model '{req.model_name}' trained successfully.",
            "metrics": metrics
        }

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"CSV file '{req.csv_path}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.post("/predict")
def predict_api(req: PredictRequest):
    try:
        preds = predict(req.model_name, req.data)
        return {"predictions": preds}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/columns/{model_name}")
def get_model_columns(model_name: str):
    if model_name not in model_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' hasn't been trained or doesn't exist."
        )
    
    return {"feature_columns": model_columns[model_name]}

@app.get("/")
def root():
    return {"message": "Linear Regression ML Service is running. Visit /docs for API interface."}


