import joblib
import pandas as pd
import pytest

from src.training import train


def test_prepare_data():
    """Prepare the data by applying preprocessing and returns 70/15/15 splits."""
    df = pd.DataFrame(
        {
            "title": ["I am happy to be there today !"] * 10 + ["I am sad !!!!"] * 10,
            "label": [1] * 10 + [0] * 10,
        }
    )
    X_train, y_train, X_val, y_val, X_test, y_test = train.prepare_data(df_cleaned=df)

    assert len(X_train) == 14
    assert len(X_val) == 3
    assert len(X_test) == 3
    assert len(y_train) == 14
    assert len(y_val) == 3
    assert len(y_test) == 3


def test_train_log_model():
    """train_model returns a fitted TF-IDF vectorizer and logistic model."""
    X_train = pd.Series(["happy calm"] * 6 + ["sad dark"] * 6)
    y_train = pd.Series([1] * 6 + [0] * 6)

    vectorizer, model = train.train_log_model(X_train, y_train)

    assert "happy" in vectorizer.vocabulary_
    assert "sad" in vectorizer.vocabulary_
    assert model.class_weight == "balanced"
    assert set(model.classes_) == {0, 1}


def test_save_artifacts(tmp_path):
    """save_artifacts persists vectorizer/model to configured paths."""
    models_dir = tmp_path / "models"
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    model_path = models_dir / "lr_model.pkl"

    vectorizer_obj = {"artifact": "vectorizer"}
    model_obj = {"artifact": "model"}

    train.save_artifacts(
        vectorizer_obj,
        model_obj,
        models_dir=models_dir,
        vectorizer_path=vectorizer_path,
        model_path=model_path,
    )

    assert vectorizer_path.exists()
    assert model_path.exists()
    assert joblib.load(vectorizer_path) == vectorizer_obj
    assert joblib.load(model_path) == model_obj


def test_train_xgboost_model():
    """train_xgboost_model returns a fitted TF-IDF vectorizer and XGBoost model."""
    pytest.importorskip("xgboost")

    X_train = pd.Series(["happy calm"] * 6 + ["sad dark"] * 6)
    y_train = pd.Series([1] * 6 + [0] * 6)

    vectorizer, model = train.train_xgboost_model(X_train, y_train)

    assert "happy" in vectorizer.vocabulary_
    assert "sad" in vectorizer.vocabulary_
    assert vectorizer.max_features == 10000
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.stop_words == "english"
    assert set(model.classes_) == {0, 1}
