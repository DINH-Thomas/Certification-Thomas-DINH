import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src.config.config import (
    DATA_DIR,
    DATA_FILENAME,
    HEADERS,
    KAGGLE_KEY,
    KAGGLE_USERNAME,
    MAX_POSTS_PER_SUBREDDIT,
    OUTPUT_PATH,
    SLEEP_BETWEEN_REQUESTS,
    SUBREDDITS,
)


def download_data_kaggle() -> None:
    """Load the dataset from the Kaggle API.
    KAGGLE_USERNAME and KAGGLE_KEY MUST BE SET IN YOUR .env FILE !!!!!!"""
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Inject credentials loaded in config into process env for kaggle package.
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY

    if not os.environ["KAGGLE_USERNAME"] or not os.environ["KAGGLE_KEY"]:
        raise ValueError("KAGGLE_USERNAME ou KAGGLE_KEY manquant dans le .env")

    # Initialiser et authentifier l'API
    api = KaggleApi()
    api.authenticate()

    expected_file = DATA_DIR / DATA_FILENAME

    if expected_file.exists():
        print("Dataset déjà téléchargé, téléchargement ignoré.")
    else:
        api.dataset_download_files(
            "rishabhkausish/reddit-depression-dataset",
            path=str(DATA_DIR),
            unzip=True,
        )
        print("Dataset téléchargé.")


def scrape_subreddit(name: str, max_posts: int = 300) -> list:
    """Scrape posts from a subreddit using Reddit's public .json feed."""
    posts = []
    after = None
    base_url = f"https://www.reddit.com/r/{name}.json"

    print(f"Scraping r/{name} ...", end="")

    while len(posts) < max_posts:
        params = {"limit": 100}
        if after:
            params["after"] = after

        try:
            response = requests.get(base_url, headers=HEADERS, params=params, timeout=15)
        except requests.RequestException as e:
            print(f" request error: {e}")
            break

        if response.status_code == 429:
            print(" rate limited, waiting 30s ...")
            time.sleep(30)
            continue
        if response.status_code != 200:
            print(f" HTTP {response.status_code}, stopping.")
            break

        data = response.json().get("data", {})
        children = data.get("children", [])

        if not children:
            break

        for child in children:
            post = child.get("data", {})
            if post.get("stickied") or post.get("author") == "[deleted]":
                continue
            body = post.get("selftext", "") or ""
            if body in ("[deleted]", "[removed]"):
                body = ""
            posts.append(
                {
                    "subreddit": name,
                    "title": post.get("title", ""),
                    "body": body,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": datetime.utcfromtimestamp(post.get("created_utc", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        after = data.get("after")
        print(f" {len(posts)}", end="", flush=True)

        if not after:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f" -> {len(posts)} posts collected")
    return posts


def all_posts_listed() -> pd.DataFrame:
    """List all scraps subreddits."""
    all_posts = []
    for subreddit in SUBREDDITS:
        posts = scrape_subreddit(subreddit, max_posts=MAX_POSTS_PER_SUBREDDIT)
        all_posts.extend(posts)
    df = pd.DataFrame(all_posts)
    return df


def save_posts_to_csv(df: pd.DataFrame, output_path: Path = OUTPUT_PATH):
    """Save the collected posts to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
