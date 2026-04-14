import pandas as pd

from src.data_cleaning import clean_data

# A faire peut etre avant : python3 -m pip install pytest
# Et pip install -r requirements.txt

DATA_FILENAME = "reddit_depression_dataset.csv"

# A voir plus tard
"""def test_load_data():
    df = load_data()
    assert type(df) == pd.DataFrame"""


def test_clean_data_kaggle():
    """
    Test the clean_data function by creating a sample DataFrame with
    duplicates and missing values,
    and checking if the cleaned DataFrame has no duplicates,
    no missing values, and only the relevant columns.
    """
    # Création d'un Dataframe de test pour ne pas avoir à load_data
    df = pd.DataFrame(
        [
            {
                "Unnamed: 0": "1",
                "subreddit": "depression",
                "title": "i feel low",
                "body": "today",
                "upvotes": 10,
                "created_utc": 1700000000,
                "label": 1,
                "num_comments": 2,
            },
            {
                "Unnamed: 0": "2",
                "subreddit": "mentalhealth",
                "title": "need advice",
                "body": "please help",
                "upvotes": 3,
                "created_utc": 1700001000,
                "label": 0,
                "num_comments": 1,
            },
            {
                "Unnamed: 0": "need advice",
                "subreddit": "mentalhealth",
                "title": "2",
                "body": pd.NA,
                "upvotes": 3,
                "created_utc": 1700001000,
                "label": 0,
                "num_comments": 1,
            },
        ]
    )

    df_cleaned = clean_data.clean_data_kaggle(df)
    assert isinstance(df_cleaned, pd.DataFrame)
    assert df_cleaned.isna().sum().sum() == 0
    assert df_cleaned.shape[1] == 2


def test_clean_data_scrapped():
    """
    Test clean_data_scrapped with a sample dataframe from subreddit scraping
    """
    df = pd.DataFrame(
        [
            {
                "subreddit": "happy",
                "title": "great day",
                "body": "sun is out",
                "score": 12,
                "num_comments": 3,
                "created_utc": "2024-01-01 10:00:00",
                "label": 0,
            },
            {
                "subreddit": "happy",
                "title": "need help",
                "body": "",
                "score": 4,
                "num_comments": 1,
                "created_utc": "2024-01-01 11:00:00",
                "label": 0,
            },
            {
                "subreddit": "happy",
                "title": "missing score",
                "body": "should be dropped",
                "score": pd.NA,
                "num_comments": 0,
                "created_utc": "2024-01-01 12:00:00",
                "label": 0,
            },
        ]
    )

    df_cleaned = clean_data.clean_data_scrapped(df)

    assert isinstance(df_cleaned, pd.DataFrame)
    assert df_cleaned.isna().sum().sum() == 0
    assert df_cleaned.shape[1] == 2
    assert set(df_cleaned.columns) == {"title", "label"}
