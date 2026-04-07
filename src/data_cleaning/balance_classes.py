import pandas as pd


# --- Balance classes to equal size ---
def balance_classes(df_dep: pd.DataFrame, df_happy: pd.DataFrame, target_per_class: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """Balance the dataset to have 50/50 labels happy and depressed"""
    n = min(len(df_dep), len(df_happy))
    if n < target_per_class:
        print(f"Balancing to {n:,} rows per class.")

    df_dep = df_dep.sample(n, random_state=random_state).reset_index(drop=True)
    df_happy = df_happy.sample(n, random_state=random_state).reset_index(drop=True)

    # Combine and shuffle
    df = pd.concat([df_dep, df_happy], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nFinal dataset shape: {df.shape}")
    print(df["label"].value_counts())
    return df


def remove_label0_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the label 0 from the kaggle dataset."""
    if "label" in df.columns:
        df = df[df["label"] != 0]
    return df
