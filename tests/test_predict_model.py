from app.predict_model import predict
from app.create_model import create_model
from app.train_model import train_model

def test_predict_correct():
    model_name = "predict_test_model"
    create_model(model_name)
    train_model(model_name, "tests/data/train.csv", drop_columns=["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"])
    
    data = [{"years_experience": 1, "skills_count": 6, "resume_score": 81}]
    preds = predict(model_name, data)
    assert isinstance(preds, list)
