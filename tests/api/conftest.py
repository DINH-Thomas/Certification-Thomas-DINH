"""Pytest configuration for API tests - setup test database before any imports."""

import os

# MUST be set BEFORE importing any fastapi/database modules
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import pytest
from fastapi.testclient import TestClient


def _fake_predict(text: str, model_type: str = "lr") -> dict:
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    supported = {"lr", "xgboost", "distilbert", "mental_roberta", "mentalbert"}
    if model_type not in supported:
        raise ValueError(f"Unsupported model_type: {model_type}")

    distress_markers = {"sad", "hopeless", "pointless", "exhausted", "depressed"}
    lowered = text.lower()
    distress_score = sum(marker in lowered for marker in distress_markers)
    probability = 0.82 if distress_score > 0 else 0.18
    label = 1 if probability >= 0.5 else 0
    return {"label": label, "probability": probability}


def _fake_explain(
    text: str,
    model_type: str = "lr",
    threshold: float = 0.05,
    max_tokens: int = 40,
) -> dict:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0.")

    prediction = _fake_predict(text, model_type)
    label = int(prediction["label"])
    probability = float(prediction["probability"])

    tokens = [tok.strip(".,!?;:\"'()[]{}") for tok in text.split() if tok.strip()]
    tokens = tokens[:max_tokens]
    word_importance = {token.lower(): 0.12 for token in tokens if token}
    colored_html = " ".join(f'<span style="color:red">{token}</span>' for token in tokens)

    if label == 1:
        display_confidence = probability
        confidence_label = "distress"
    else:
        display_confidence = 1.0 - probability
        confidence_label = "no_distress"

    return {
        "label": label,
        "probability": probability,
        "display_confidence": display_confidence,
        "confidence_label": confidence_label,
        "risk_level": "high" if probability >= 0.66 else "medium" if probability >= 0.33 else "low",
        "colored_html": colored_html,
        "word_importance": word_importance,
        "note": None,
    }


def _fake_stats() -> dict:
    return {
        "total_predictions": 0,
        "distress_count": 0,
        "no_distress_count": 0,
        "risk_level_counts": {"low": 0, "medium": 0, "high": 0},
        "model_usage": {},
        "predictions_by_day": [],
        "avg_confidence": 0.0,
        "distress_by_model": {},
    }


@pytest.fixture
def client(monkeypatch):
    """FastAPI test client with SQLite in-memory database."""
    # Now safe to import after DATABASE_URL is set
    from src.api.database import Base, engine, init_db
    from src.api import main

    monkeypatch.setattr(main, "ensure_models", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "init_db", lambda: None)
    monkeypatch.setattr(main, "predict_service", _fake_predict)
    monkeypatch.setattr(main, "explain_service", _fake_explain)
    monkeypatch.setattr(main, "get_stats", _fake_stats)
    monkeypatch.setattr(main, "log_prediction", lambda *args, **kwargs: None)

    # Initialize database tables
    Base.metadata.create_all(engine)
    init_db()

    return TestClient(main.app)
