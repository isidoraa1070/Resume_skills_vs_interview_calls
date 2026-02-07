from app.train_model import train_model
from app.create_model import create_model


def test_train_model_metrics():
    model_name = "train_test_model"
    create_model(model_name)
    metrics, X = train_model(
        model_name=model_name,
        csv_path="tests/data/train.csv",
        drop_columns=["candidate_id", "github_portfolio", "degree","certifications","projects_count","internship"]
    )
    assert "mse" in metrics
    assert "r2" in metrics
