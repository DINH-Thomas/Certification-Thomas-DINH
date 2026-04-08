import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import config
from src.training.preprocess import preprocess_text


def _load_cleaned_dataframe(cleaned_csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the default cleaned dataset used for training."""
    csv_path = cleaned_csv_path or (config.PROJECT_ROOT / "data" / "cleaned" / "balanced_30k_dataset.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {csv_path}. Run the cleaning pipeline first or pass df_cleaned explicitly.")
    return pd.read_csv(csv_path)


def prepare_data(df_cleaned: Optional[pd.DataFrame] = None, cleaned_csv_path: Optional[Path] = None) -> tuple:
    """
    Preprocess the text,
    and split it into train, validation, and test sets.
    """
    if df_cleaned is None:
        df_cleaned = _load_cleaned_dataframe(cleaned_csv_path)

    df_cleaned["title"] = df_cleaned["title"].apply(preprocess_text)
    X = df_cleaned["title"]
    y = df_cleaned["label"]
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50, random_state=42, stratify=y_test_val)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_log_model(X_train, y_train) -> tuple:
    """Train a logistic regression model using TF-IDF features.
    - We use a TfidfVectorizer to convert the text data into TF-IDF features,
        with a maximum of 50,000 features, unigrams and bigrams, and a minimum document frequency of 5.
    - We train a LogisticRegression model with balanced class weights
        and a maximum of 1000 iterations.
    - The function returns the fitted vectorizer and model"""
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    X_train = vectorizer.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return vectorizer, model


def train_xgboost_model(X_train, y_train, model_factory=None) -> tuple:
    """Train an XGBoost model using the same setup as the exploratory notebook.
    - Uses TF-IDF with 10k features, unigrams+bigrams and English stop-words.
    - Uses tuned XGBoost hyperparameters from grid search.
    - Optional model_factory is provided for lightweight tests without importing xgboost.
    """
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    if model_factory is None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required to train the XGBoost model. Install it with `pip install xgboost`.") from exc

        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4.2,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        model = model_factory()

    model.fit(X_train_tfidf, y_train)
    return vectorizer, model


def save_artifacts(vectorizer, model, models_dir=None, vectorizer_path=None, model_path=None):
    """Save the trained model and vectorizer to disk using joblib.
    Optional paths make the function easy to test without monkeypatching."""
    if models_dir is None:
        models_dir = config.MODELS_DIR
    if vectorizer_path is None:
        vectorizer_path = config.VECTORIZER_PATH
    if model_path is None:
        model_path = config.LR_MODEL_PATH

    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    lr_vectorizer, lr_model = train_log_model(X_train, y_train)
    save_artifacts(lr_vectorizer, lr_model)

    xgb_vectorizer, xgb_model = train_xgboost_model(X_train, y_train)
    save_artifacts(
        xgb_vectorizer,
        xgb_model,
        vectorizer_path=config.XGBOOST_VECTORIZER_PATH,
        model_path=config.XGBOOST_MODEL_PATH,
    )
