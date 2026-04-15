"""End-to-end data + training pipeline.

This module orchestrates, in order:
1) Kaggle dataset download
2) Reddit scraping for positive subreddits
3) Data cleaning + class balancing
4) Logistic Regression and XGBoost training
5) Metrics computation (accuracy, precision, recall, f1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import config
from src.data_cleaning import download_data
from src.data_cleaning.balance_classes import balance_classes, remove_label0_kaggle
from src.data_cleaning.clean_data import add_label0, clean_data_kaggle, clean_data_scrapped
from src.training.evaluate import evaluate
from src.training.train import prepare_data, save_artifacts, train_log_model, train_xgboost_model


def _format_metrics(metrics: dict) -> str:
    """Render the metric subset requested by the user."""
    return (
        f"accuracy={metrics['accuracy']:.4f} | precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | f1={metrics['f1_score']:.4f}"
    )


def _persist_balanced_dataset(df_balanced: pd.DataFrame, output_path: Path) -> None:
    """Persist the final cleaned and balanced dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)


def run_pipeline(max_posts_per_subreddit: int | None = None) -> dict:
    """Run the complete pipeline and return metrics for both models."""
    print("Step 1/7 - Download Kaggle data")
    download_data.download_data_kaggle()

    kaggle_csv_path = config.DATA_DIR / config.DATA_FILENAME
    if not kaggle_csv_path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found at {kaggle_csv_path}")

    print("Step 2/7 - Load + clean Kaggle data")
    df_kaggle = pd.read_csv(kaggle_csv_path)
    df_kaggle = remove_label0_kaggle(df_kaggle)
    df_dep_clean = clean_data_kaggle(df_kaggle)

    print("Step 3/7 - Scrape Reddit positive subreddits")
    if max_posts_per_subreddit is not None:
        original_max_posts = download_data.MAX_POSTS_PER_SUBREDDIT
        download_data.MAX_POSTS_PER_SUBREDDIT = max_posts_per_subreddit
    else:
        original_max_posts = None

    try:
        df_scraped = download_data.all_posts_listed()
    finally:
        if original_max_posts is not None:
            download_data.MAX_POSTS_PER_SUBREDDIT = original_max_posts

    if df_scraped.empty:
        raise ValueError("Scraping returned no posts. Check Reddit headers/network and retry.")

    raw_scraped_path = config.PROJECT_ROOT / "data" / "raw" / "happiness_reddit.csv"
    download_data.save_posts_to_csv(df_scraped, output_path=raw_scraped_path)

    print("Step 4/7 - Clean scraped data + balance classes")
    df_scraped = add_label0(df_scraped)
    df_happy_clean = clean_data_scrapped(df_scraped)
    df_balanced = balance_classes(df_dep_clean, df_happy_clean)

    final_dataset_path = config.PROJECT_ROOT / "data" / "cleaned" / "balanced_30k_dataset.csv"
    _persist_balanced_dataset(df_balanced, final_dataset_path)

    print("Step 5/7 - Prepare train/val/test splits")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(df_cleaned=df_balanced)
    del X_val, y_val

    print("Step 6/7 - Train + evaluate Logistic Regression")
    lr_vectorizer, lr_model = train_log_model(X_train, y_train)
    save_artifacts(lr_vectorizer, lr_model)
    lr_metrics = evaluate(lr_model, lr_vectorizer, X_test, y_test)

    print("Step 7/7 - Train + evaluate XGBoost")
    xgb_vectorizer, xgb_model = train_xgboost_model(X_train, y_train)
    save_artifacts(
        xgb_vectorizer,
        xgb_model,
        vectorizer_path=config.XGBOOST_VECTORIZER_PATH,
        model_path=config.XGBOOST_MODEL_PATH,
    )
    xgb_metrics = evaluate(xgb_model, xgb_vectorizer, X_test, y_test)

    print("\nPipeline complete.")
    print(f"Logistic Regression -> {_format_metrics(lr_metrics)}")
    print(f"XGBoost             -> {_format_metrics(xgb_metrics)}")
    print(f"Balanced dataset saved at: {final_dataset_path}")

    return {
        "dataset_path": str(final_dataset_path),
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: Kaggle download + Reddit scraping + cleaning + balancing + LR/XGBoost training + metrics."
    )
    parser.add_argument(
        "--max-posts-per-subreddit",
        type=int,
        default=None,
        help="Override maximum number of scraped posts per subreddit for this run.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for full pipeline execution."""
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(max_posts_per_subreddit=args.max_posts_per_subreddit)


if __name__ == "__main__":
    main()
