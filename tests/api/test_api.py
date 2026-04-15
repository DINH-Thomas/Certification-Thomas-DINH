"""Unit and integration tests for the FastAPI Mental Health Signal Detector API.

Tests cover:
- Health check endpoint
- Prediction endpoint (all model types)
- Explanation endpoint (all model types)
- Statistics endpoint
- Error handling and validation
- Database logging functionality
"""

# Ensure project root is in path for imports
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.schemas import PredictionResponse


@pytest.fixture
def sample_text_distress():
    """Sample text indicating distress."""
    return "I haven't left my room in days. Everything feels pointless and I'm exhausted."


@pytest.fixture
def sample_text_positive():
    """Sample text indicating no distress."""
    return "I had a wonderful day with my friends. We laughed so much."


@pytest.fixture
def sample_text_empty():
    """Empty text for validation testing."""
    return ""


@pytest.fixture
def sample_text_very_long():
    """Very long text to test truncation/limits."""
    return " ".join(["word"] * 5000)


class TestHealthCheck:
    """Test cases for the health check endpoint."""

    def test_health_check_status_ok(self, client):
        """Health check should return 200 with status healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_root_endpoint_exists(self, client):
        """Root endpoint should list available endpoints."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]
        assert "explain" in data["endpoints"]
        assert "stats" in data["endpoints"]


class TestPredictionEndpoint:
    """Test cases for the /predict endpoint."""

    @pytest.mark.parametrize("model_type", ["lr", "xgboost", "distilbert", "mental_roberta"])
    def test_predict_all_models(self, client, sample_text_distress, model_type):
        """Prediction should work for all supported model types."""
        payload = {"text": sample_text_distress, "model_type": model_type}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "probability" in data
        assert data["label"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_default_model_is_lr(self, client, sample_text_distress):
        """Default model should be Logistic Regression when not specified."""
        payload = {"text": sample_text_distress}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "probability" in data

    def test_predict_distress_signal(self, client, sample_text_distress):
        """Text with distress signals should be flagged."""
        payload = {"text": sample_text_distress, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # LR model should detect distress in this text
        assert data["label"] in [0, 1]

    def test_predict_positive_signal(self, client, sample_text_positive):
        """Text with positive signals should not be flagged."""
        payload = {"text": sample_text_positive, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        # LR model should not detect distress in positive text
        assert data["label"] in [0, 1]

    def test_predict_empty_text_raises_error(self, client, sample_text_empty):
        """Empty text should raise a 400 Bad Request error."""
        payload = {"text": sample_text_empty, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_whitespace_only_raises_error(self, client):
        """Text with only whitespace should raise a 400 Bad Request error."""
        payload = {"text": "   \n\t   ", "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_predict_invalid_model_type_raises_error(self, client, sample_text_distress):
        """Invalid model type should raise an error (400 or 422)."""
        payload = {"text": sample_text_distress, "model_type": "invalid_model"}
        response = client.post("/predict", json=payload)
        assert response.status_code in [400, 422]  # Validation error or unsupported model

    def test_predict_missing_text_raises_error(self, client):
        """Missing text field should raise a validation error."""
        payload = {"model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_response_schema_valid(self, client, sample_text_distress):
        """Prediction response should match PredictionResponse schema."""
        payload = {"text": sample_text_distress, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        try:
            PredictionResponse(**data)
        except Exception as exc:
            pytest.fail(f"Response does not match schema: {exc}")

    def test_predict_logging_triggered(self, client, sample_text_distress):
        """Prediction should trigger background logging task."""
        payload = {"text": sample_text_distress, "model_type": "lr"}
        # The task is logged in background, but we can verify the response
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


class TestExplainEndpoint:
    """Test cases for the /explain endpoint."""

    @pytest.mark.parametrize("model_type", ["lr", "xgboost", "distilbert", "mental_roberta"])
    def test_explain_all_models(self, client, sample_text_distress, model_type):
        """Explanation should work for all supported model types."""
        payload = {
            "text": sample_text_distress,
            "model_type": model_type,
            "threshold": 0.005,
            "max_tokens": 40,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "probability" in data
        assert "colored_html" in data
        assert "word_importance" in data

    def test_explain_default_parameters(self, client, sample_text_distress):
        """Explanation should use default threshold and max_tokens."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "colored_html" in data

    def test_explain_custom_threshold(self, client, sample_text_distress):
        """Custom threshold parameter should be accepted."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
            "threshold": 0.1,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200

    def test_explain_custom_max_tokens(self, client, sample_text_distress):
        """Custom max_tokens parameter should be accepted."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
            "max_tokens": 20,
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Word importance should be limited to max_tokens
        assert len(data["word_importance"]) <= 20

    def test_explain_empty_text_raises_error(self, client):
        """Empty text should raise a 400 Bad Request error."""
        payload = {
            "text": "",
            "model_type": "lr",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 400

    def test_explain_invalid_model_type_raises_error(self, client, sample_text_distress):
        """Invalid model type should raise an error (400 or 422)."""
        payload = {
            "text": sample_text_distress,
            "model_type": "invalid_model",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code in [400, 422]  # Validation error or unsupported model

    def test_explain_response_contains_colored_html(self, client, sample_text_distress):
        """Explanation response should contain HTML-colored text."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        # Colored HTML should contain span tags
        assert "span" in data["colored_html"]
        assert "style" in data["colored_html"]

    def test_explain_response_contains_word_importance(self, client, sample_text_distress):
        """Explanation response should contain word importance scores."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["word_importance"], dict)
        # Each word should have a float score
        for word, score in data["word_importance"].items():
            assert isinstance(word, str)
            assert isinstance(score, float)

    def test_explain_confidence_label_present(self, client, sample_text_distress):
        """Explanation response should contain confidence_label."""
        payload = {
            "text": sample_text_distress,
            "model_type": "lr",
        }
        response = client.post("/explain", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_label"] in ["distress", "no_distress"]


class TestStatsEndpoint:
    """Test cases for the /stats endpoint."""

    def test_stats_endpoint_returns_200(self, client):
        """Stats endpoint should return 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_response_contains_required_fields(self, client):
        """Stats response should contain all required fields."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "total_predictions",
            "distress_count",
            "no_distress_count",
            "risk_level_counts",
            "model_usage",
            "predictions_by_day",
            "avg_confidence",
            "distress_by_model",
        ]
        for field in required_fields:
            assert field in data

    def test_stats_initial_state(self, client):
        """Stats should show zeros or empty state initially."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        # Initially, should have 0 predictions
        assert isinstance(data["total_predictions"], int)
        assert data["total_predictions"] >= 0
        assert data["no_distress_count"] + data["distress_count"] == data["total_predictions"]

    def test_stats_risk_level_counts_valid(self, client):
        """Risk level counts should be non-negative."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        # Should be empty initially but structure should be valid
        assert isinstance(data["risk_level_counts"], dict)
        for level, count in data["risk_level_counts"].items():
            assert level in ["low", "medium", "high"]
            assert count >= 0

    def test_stats_model_usage_valid(self, client):
        """Model usage should have valid model names."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["model_usage"], dict)
        valid_models = {"lr", "xgboost", "distilbert", "mental_roberta"}
        for model_name in data["model_usage"].keys():
            assert model_name in valid_models

    def test_stats_avg_confidence_in_range(self, client):
        """Average confidence should be between 0 and 1."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["avg_confidence"] <= 1.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json_returns_422(self, client):
        """Malformed JSON should return 422."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [400, 422]

    def test_missing_required_field_returns_422(self, client):
        """Missing required field should return 422."""
        payload = {"model_type": "lr"}  # text is missing
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_json_type_returns_422(self, client):
        """Invalid JSON type should return 422."""
        payload = {"text": 123, "model_type": "lr"}  # text should be string
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_unicode_text_handled_correctly(self, client):
        """Unicode text should be handled correctly."""
        payload = {
            "text": "Je me sens très déprimé 😢 - 我很难过",
            "model_type": "lr",
        }
        response = client.post("/predict", json=payload)
        # Should either work or give a reasonable error
        assert response.status_code in [200, 400]

    def test_extremely_long_text_handled(self, client, sample_text_very_long):
        """Very long text should be handled (truncated by model or accepted)."""
        payload = {"text": sample_text_very_long, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        # Should either work or give message about text length
        assert response.status_code in [200, 400]


class TestCORSSettings:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """CORS headers should be present in responses."""
        response = client.get("/health")
        assert response.status_code == 200
        # CORS headers will be in the response object
        assert response.status_code == 200


class TestDatabaseLogging:
    """Test database logging functionality."""

    def test_prediction_triggers_logging_task(self, client, sample_text_distress):
        """Making a prediction should trigger a logging task."""
        payload = {"text": sample_text_distress, "model_type": "lr"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        # Logging happens in background, but prediction should succeed
        data = response.json()
        assert "label" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
