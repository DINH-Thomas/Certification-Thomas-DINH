"""Shared configuration values."""

import json
import os

import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
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

