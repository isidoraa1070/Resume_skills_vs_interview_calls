from fastapi.testclient import TestClient
from app.app import app
from app.create_model import create_model, models
from app.train_model import train_model
from app.predict_model import predict
import pytest
from utils.save_model_columns import model_columns  # global dict to store feature columns for each model

client = TestClient(app)

# Test da create_model radi
def test_create_model_api_success():
    response = client.post("/create_model", json={"model_name": "api_model"})
    assert response.status_code == 200
    assert "kreiran" in response.json()["message"]

# Test da create_model baci grešku za duplikat
def test_create_model_api_duplicate():
    client.post("/create_model", json={"model_name": "dup_model"})
    response = client.post("/create_model", json={"model_name": "dup_model"})
    assert response.status_code == 400

def test_train_model_api():
    # kreiramo model
    client.post("/create_model", json={"model_name": "train_model"})

    # treniramo model
    response = client.post("/train_model", json={
        "model_name": "train_model",
        "csv_path": "tests/data/train.csv",
        "drop_columns": ["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"]
    })

    assert response.status_code == 200
    resp = response.json()
    assert "mse" in resp["metrics"]
    assert "r2" in resp["metrics"]
    assert "trained successfully" in resp["message"]


def test_predict_correct():
    model_name = "predict_test_model_correct"
    create_model(model_name)
    train_model(model_name, "tests/data/train.csv", drop_columns=["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"])
    
    data = [{"years_experience": 1, "skills_count": 6, "resume_score": 81}]
    preds = predict(model_name, data)
    assert isinstance(preds, list)

def test_predict_missing_column():
    model_name = "predict_test_model_missing"
    create_model(model_name)
    train_model(model_name, "tests/data/train.csv", drop_columns=["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"])
    
    # nedostaje kolona
    data = [{"years_experience": 1, "skills_count": 6}]
    with pytest.raises(ValueError):
        predict(model_name, data)

def test_predict_extra_column():
    model_name = "predict_test_model_extra_column"
    create_model(model_name)
    train_model(model_name, "tests/data/train.csv", drop_columns=["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"])
    
    # višak kolona
    data = [{"years_experience": 1, "skills_count": 6, "resume_score": 81, "extra_col": 123}]
    with pytest.raises(ValueError):
        predict(model_name, data)


