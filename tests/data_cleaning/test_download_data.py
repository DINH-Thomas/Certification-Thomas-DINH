import pandas as pd

from src.data_cleaning import download_data


def test_scrape_subreddit():
    """Scrap 1 Reddit post and check if it isn't an empty list."""
    posts = download_data.scrape_subreddit("happy", max_posts=1)

    assert isinstance(posts, list)
    assert len(posts) > 0
    assert posts[0]["subreddit"] == "happy"
    assert posts[0]["title"] != ""


def test_all_posts_listed():
    """Download the post and convert it to a DataFrame, then check if it's not empty."""
    original_subreddits = download_data.SUBREDDITS
    original_max_posts = download_data.MAX_POSTS_PER_SUBREDDIT

    download_data.SUBREDDITS = ["happy"]
    download_data.MAX_POSTS_PER_SUBREDDIT = 1

    try:
        df = download_data.all_posts_listed()
    finally:
        download_data.SUBREDDITS = original_subreddits
        download_data.MAX_POSTS_PER_SUBREDDIT = original_max_posts

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
