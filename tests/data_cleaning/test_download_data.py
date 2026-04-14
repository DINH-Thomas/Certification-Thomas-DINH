import pandas as pd

from src.data_cleaning import download_data


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _fake_reddit_payload():
    return {
        "data": {
            "children": [
                {
                    "data": {
                        "subreddit": "happy",
                        "title": "A good day",
                        "selftext": "Feeling great",
                        "score": 10,
                        "num_comments": 2,
                        "created_utc": 1700000000,
                        "stickied": False,
                        "author": "someone",
                    }
                }
            ],
            "after": None,
        }
    }


def _mock_get(*args, **kwargs):
    return _FakeResponse(_fake_reddit_payload())


def test_scrape_subreddit(monkeypatch):
    """Scrap 1 Reddit post and check if it isn't an empty list."""
    monkeypatch.setattr(download_data.requests, "get", _mock_get)
    monkeypatch.setattr(download_data.time, "sleep", lambda *args, **kwargs: None)

    posts = download_data.scrape_subreddit("happy", max_posts=1)

    assert isinstance(posts, list)
    assert len(posts) > 0
    assert posts[0]["subreddit"] == "happy"
    assert posts[0]["title"] != ""


def test_all_posts_listed(monkeypatch):
    """Download the post and convert it to a DataFrame, then check if it's not empty."""
    original_subreddits = download_data.SUBREDDITS
    original_max_posts = download_data.MAX_POSTS_PER_SUBREDDIT

    download_data.SUBREDDITS = ["happy"]
    download_data.MAX_POSTS_PER_SUBREDDIT = 1
    monkeypatch.setattr(download_data.requests, "get", _mock_get)
    monkeypatch.setattr(download_data.time, "sleep", lambda *args, **kwargs: None)

    try:
        df = download_data.all_posts_listed()
    finally:
        download_data.SUBREDDITS = original_subreddits
        download_data.MAX_POSTS_PER_SUBREDDIT = original_max_posts

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
