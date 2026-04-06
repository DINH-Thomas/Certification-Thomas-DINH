import pandas as pd


def clean_data_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by removing duplicates, handling missing values,
    and balancing classes."""
    # Remove duplicates
    df = df.drop_duplicates()
    # Remove na_values for those who haven't not a lot of na
    df = df.dropna(
        subset=["Unnamed: 0", "subreddit", "title", "upvotes", "created_utc", "label"],
        axis=0,
    )
    # Inversion de certaines valeurs de Unnamed: 0 et title
    unnamed_is_numeric = pd.to_numeric(df["Unnamed: 0"], errors="coerce").notna()
    title_is_numeric = pd.to_numeric(df["title"], errors="coerce").notna()
    mask = (~unnamed_is_numeric) & title_is_numeric

    # 3) Inversion des valeurs entre les 2 colonnes sur ces lignes
    df.loc[mask, ["Unnamed: 0", "title"]] = df.loc[mask, ["title", "Unnamed: 0"]].to_numpy()
    df.loc[mask, ["Unnamed: 0", "title"]] = df.loc[mask, ["title", "Unnamed: 0"]].to_numpy()
    # On drop les colonnes qui ne sont pas utiles
    # On va fusionner les colonnes de texte body et title
    sub_df = df[["title", "body"]]
    text = sub_df["title"].fillna("") + " " + sub_df["body"].fillna("")
    df["title"] = text
    df.drop(
        columns=[
            "Unnamed: 0",
            "body",
            "subreddit",
            "upvotes",
            "created_utc",
            "num_comments",
        ],
        inplace=True,
    )

    df.dropna(inplace=True)
    return df


def clean_data_scrapped(df: pd.DataFrame) -> pd.DataFrame:
    """Change some columns names but it the same function."""
    # Remove duplicates
    df = df.drop_duplicates()
    # Remove na_values for those who haven't not a lot of na
    df = df.dropna(
        subset=["subreddit", "title", "score", "num_comments", "created_utc", "label"],
        axis=0,
    )
    # On drop les colonnes qui ne sont pas utiles
    # On va fusionner les colonnes de texte body et title
    sub_df = df[["title", "body"]]
    text = sub_df["title"].fillna("") + " " + sub_df["body"].fillna("")
    df["title"] = text
    df.drop(
        columns=[
            "body",
            "subreddit",
            "score",
            "created_utc",
            "num_comments",
        ],
        inplace=True,
    )

    df.dropna(inplace=True)
    return df
