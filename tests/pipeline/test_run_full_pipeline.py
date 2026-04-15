import pandas as pd

from src.pipeline import run_full_pipeline


def test_run_pipeline_happy_path(monkeypatch):
    """
    Test the full pipeline execution with typical data.
    """
    kaggle_df = pd.DataFrame(
        {
            "Unnamed: 0": ["1", "2"],
            "subreddit": ["depression", "depression"],
            "title": ["sad post", "still sad"],
            "body": ["text", "text"],
            "upvotes": [10, 12],
            "created_utc": [1700000000, 1700000100],
            "label": [1, 1],
            "num_comments": [1, 2],
        }
    )
    scraped_df = pd.DataFrame(
        {
            "subreddit": ["happy", "joy"],
            "title": ["great day", "feeling good"],
            "body": ["sunny", "calm"],
            "score": [4, 5],
            "num_comments": [1, 1],
            "created_utc": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        }
    )
    cleaned_dep = pd.DataFrame({"title": ["sad post", "still sad"], "label": [1, 1]})
    cleaned_happy = pd.DataFrame({"title": ["great day", "feeling good"], "label": [0, 0]})
    balanced_df = pd.DataFrame(
        {
            "title": ["sad post", "still sad", "great day", "feeling good"],
            "label": [1, 1, 0, 0],
        }
    )

    monkeypatch.setattr(run_full_pipeline.download_data, "download_data_kaggle", lambda: None)
    monkeypatch.setattr(run_full_pipeline.pd, "read_csv", lambda _: kaggle_df)
    monkeypatch.setattr(run_full_pipeline, "remove_label0_kaggle", lambda df: df)
    monkeypatch.setattr(run_full_pipeline, "clean_data_kaggle", lambda df: cleaned_dep)
    monkeypatch.setattr(run_full_pipeline.download_data, "all_posts_listed", lambda: scraped_df)
    monkeypatch.setattr(run_full_pipeline.download_data, "save_posts_to_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_pipeline, "add_label0", lambda df: df.assign(label=0))
    monkeypatch.setattr(run_full_pipeline, "clean_data_scrapped", lambda df: cleaned_happy)
    monkeypatch.setattr(run_full_pipeline, "balance_classes", lambda a, b: balanced_df)
    monkeypatch.setattr(run_full_pipeline, "_persist_balanced_dataset", lambda *args, **kwargs: None)

    X_train = pd.Series(["a", "b"])
    y_train = pd.Series([0, 1])
    X_test = pd.Series(["c", "d"])
    y_test = pd.Series([0, 1])
    monkeypatch.setattr(
        run_full_pipeline,
        "prepare_data",
        lambda df_cleaned: (X_train, y_train, pd.Series(["v"]), pd.Series([1]), X_test, y_test),
    )

    lr_vectorizer, lr_model = object(), object()
    xgb_vectorizer, xgb_model = object(), object()
    monkeypatch.setattr(run_full_pipeline, "train_log_model", lambda X, y: (lr_vectorizer, lr_model))
    monkeypatch.setattr(run_full_pipeline, "train_xgboost_model", lambda X, y: (xgb_vectorizer, xgb_model))
    monkeypatch.setattr(run_full_pipeline, "save_artifacts", lambda *args, **kwargs: None)

    def fake_eval(model, vectorizer, X, y):
        """Return fake evaluation metrics based on the model type."""
        if model is lr_model:
            return {"accuracy": 0.9, "precision": 0.91, "recall": 0.92, "f1_score": 0.93, "classification_report": "lr"}
        return {"accuracy": 0.8, "precision": 0.81, "recall": 0.82, "f1_score": 0.83, "classification_report": "xgb"}

    monkeypatch.setattr(run_full_pipeline, "evaluate", fake_eval)

    result = run_full_pipeline.run_pipeline()

    assert result["dataset_path"].endswith("data/cleaned/balanced_30k_dataset.csv")
    assert result["logistic_regression"]["accuracy"] == 0.9
    assert result["xgboost"]["f1_score"] == 0.83


def test_run_pipeline_overrides_and_restores_max_posts(monkeypatch):
    """Test that run_pipeline temporarily overrides MAX_POSTS_PER_SUBREDDIT and restores it after execution."""
    original_value = run_full_pipeline.download_data.MAX_POSTS_PER_SUBREDDIT

    monkeypatch.setattr(run_full_pipeline.download_data, "download_data_kaggle", lambda: None)
    monkeypatch.setattr(
        run_full_pipeline.pd,
        "read_csv",
        lambda _: pd.DataFrame(
            {
                "Unnamed: 0": ["1"],
                "subreddit": ["depression"],
                "title": ["sad post"],
                "body": ["text"],
                "upvotes": [10],
                "created_utc": [1700000000],
                "label": [1],
                "num_comments": [1],
            }
        ),
    )
    monkeypatch.setattr(run_full_pipeline, "remove_label0_kaggle", lambda df: df)
    monkeypatch.setattr(run_full_pipeline, "clean_data_kaggle", lambda df: pd.DataFrame({"title": ["sad"], "label": [1]}))

    def fake_scrape():
        assert run_full_pipeline.download_data.MAX_POSTS_PER_SUBREDDIT == 12
        return pd.DataFrame(
            {
                "subreddit": ["happy"],
                "title": ["great day"],
                "body": ["sunny"],
                "score": [4],
                "num_comments": [1],
                "created_utc": ["2024-01-01 10:00:00"],
            }
        )

    monkeypatch.setattr(run_full_pipeline.download_data, "all_posts_listed", fake_scrape)
    monkeypatch.setattr(run_full_pipeline.download_data, "save_posts_to_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_pipeline, "add_label0", lambda df: df.assign(label=0))
    monkeypatch.setattr(run_full_pipeline, "clean_data_scrapped", lambda df: pd.DataFrame({"title": ["great day"], "label": [0]}))
    monkeypatch.setattr(
        run_full_pipeline,
        "balance_classes",
        lambda a, b: pd.DataFrame({"title": ["sad", "great day"], "label": [1, 0]}),
    )
    monkeypatch.setattr(run_full_pipeline, "_persist_balanced_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_full_pipeline,
        "prepare_data",
        lambda df_cleaned: (
            pd.Series(["sad", "great"]),
            pd.Series([1, 0]),
            pd.Series(["val"]),
            pd.Series([1]),
            pd.Series(["test"]),
            pd.Series([1]),
        ),
    )
    monkeypatch.setattr(run_full_pipeline, "train_log_model", lambda X, y: (object(), object()))
    monkeypatch.setattr(run_full_pipeline, "train_xgboost_model", lambda X, y: (object(), object()))
    monkeypatch.setattr(run_full_pipeline, "save_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_full_pipeline,
        "evaluate",
        lambda *args, **kwargs: {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0, "classification_report": "ok"},
    )

    run_full_pipeline.run_pipeline(max_posts_per_subreddit=12)

    assert run_full_pipeline.download_data.MAX_POSTS_PER_SUBREDDIT == original_value


def test_run_pipeline_raises_on_empty_scraping(monkeypatch):
    """
    Test that run_pipeline raises a ValueError when scraping returns an empty dataframe."""
    monkeypatch.setattr(run_full_pipeline.download_data, "download_data_kaggle", lambda: None)
    monkeypatch.setattr(
        run_full_pipeline.pd,
        "read_csv",
        lambda _: pd.DataFrame(
            {
                "Unnamed: 0": ["1"],
                "subreddit": ["depression"],
                "title": ["sad post"],
                "body": ["text"],
                "upvotes": [10],
                "created_utc": [1700000000],
                "label": [1],
                "num_comments": [1],
            }
        ),
    )
    monkeypatch.setattr(run_full_pipeline, "remove_label0_kaggle", lambda df: df)
    monkeypatch.setattr(run_full_pipeline, "clean_data_kaggle", lambda df: pd.DataFrame({"title": ["sad"], "label": [1]}))
    monkeypatch.setattr(run_full_pipeline.download_data, "all_posts_listed", lambda: pd.DataFrame())

    try:
        run_full_pipeline.run_pipeline()
    except ValueError as exc:
        assert "Scraping returned no posts" in str(exc)
    else:
        raise AssertionError("run_pipeline should fail when scraping returns an empty dataframe")
