"""Shared configuration values."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILENAME = "reddit_depression_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
# Keep LR model aligned with tfidf_vectorizer.pkl (both 50,000 features).
LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
DISTILBERT_MODEL_HF_PATH = MODELS_DIR / "distilbert_hf"
MENTAL_ROBERTA_HF_PATH = MODELS_DIR / "mental_roberta_hf"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgb_depression_classifier.pkl"
XGBOOST_VECTORIZER_PATH = MODELS_DIR / "xgb_tfidf_vectorizer.pkl"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / "my.env", override=True)

# CHANGE/ADD NAMES FOR ALL SUBREDDITS YOU WANT TO SCRAPE
SUBREDDITS = [
    # Original positive subreddits
    "happy",
    "Happiness",
    "joy",
    "aww",
    "wholesomememes",
    # Additional anti-depression / uplifting subreddits
    "UpliftingNews",
    "GetMotivated",
    "CasualConversation",
    "Mindfulness",
    "gratitude",
    "selfimprovement",
    "mildlyinteresting",
    "Wellbeing",
    "funny",
    "todayilearned",
]
MAX_POSTS_PER_SUBREDDIT = 3000  # increased to target ~15k total
SLEEP_BETWEEN_REQUESTS = 1  # seconds — be polite to Reddit
OUTPUT_PATH = Path("../data/raw/happiness_reddit.csv")
OUTPUT_PATH_PROCESSED = Path("../data/cleaned/balanced_30k_dataset.csv")

# Environment-backed runtime settings
GDRIVE_MODEL_FOLDER_ID = os.getenv("GDRIVE_MODEL_FOLDER_ID", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL_LOCAL = os.getenv("API_URL_LOCAL", "http://127.0.0.1:8000")
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")


def _build_headers() -> dict:
    """Build request headers from env, always returning a dict."""
    raw_headers = os.getenv("HEADERS")
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            # If HEADERS isn't valid JSON, treat it as a plain user-agent value.
            return {"User-Agent": raw_headers}

    user_agent = os.getenv("USER_AGENT", "CertificationBot/1.0")
    return {"User-Agent": user_agent}


# YOU HAVE TO SET HEADERS IN YOUR .env FILE FIRST, you can type "What is my user agent on Google" and copy paste it.
HEADERS = _build_headers()
