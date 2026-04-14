from unittest.mock import Mock

import numpy as np

from src.training import predict as predict_module


def test_lr_predict():
    """lr_predict returns dict with label (0/1) and probability (0-1)."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    vectorizer = Mock()
    vectorizer.transform.return_value = np.array([[1.0]])

    result = predict_module.lr_predict(model, vectorizer, "happy day")

    assert result["label"] == 1
    assert result["probability"] == 0.7


def test_distilbert_predict():
    """distilbert_predict returns dict with label (0/1) and probability (0-1)."""
    from types import SimpleNamespace

    import torch

    model = Mock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    model.eval.return_value = None
    model.return_value = SimpleNamespace(logits=torch.tensor([[0.5, 1.5]]))

    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    result = predict_module.distilbert_predict(model, "happy day", tokenizer=tokenizer)

    assert "label" in result
    assert "probability" in result
    assert result["label"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0


def test_xgboost_predict():
    """xgboost_predict returns dict with label (0/1) and probability (0-1)."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.6, 0.4]])
    vectorizer = Mock()
    vectorizer.transform.return_value = np.array([[1.0, 2.0]])

    result = predict_module.xgboost_predict(model, vectorizer, "sad dark mood")

    assert result["label"] == 0
    assert result["probability"] == 0.4


def test_mental_roberta_predict():
    """mental_roberta_predict returns dict with label (0/1) and probability (0-1)."""
    from types import SimpleNamespace

    import torch

    model = Mock()
    model.parameters.return_value = iter([torch.tensor([1.0])])
    model.eval.return_value = None
    model.return_value = SimpleNamespace(logits=torch.tensor([[1.0, 0.2]]))

    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    result = predict_module.mental_roberta_predict(model, "anxious mood", tokenizer=tokenizer)

    assert "label" in result
    assert "probability" in result
    assert result["label"] in [0, 1]
    assert 0.0 <= result["probability"] <= 1.0
