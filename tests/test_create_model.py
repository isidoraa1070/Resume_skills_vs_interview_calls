# tests/test_create_model.py
from app.create_model import create_model, models
import pytest

def test_create_model_success():
    model_name = "unit_test_model"
    create_model(model_name)
    assert model_name in models

def test_create_model_duplicate():
    model_name = "dup_model_another"
    create_model(model_name)
    with pytest.raises(ValueError):
        create_model(model_name)
