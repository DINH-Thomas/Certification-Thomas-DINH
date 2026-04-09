import pandas as pd

from src.data_cleaning import balance_classes


def test_balance_classes():
    """Test the balance_classes function by creating two sample DataFrames with different sizes,
    and checking if the balanced DataFrame has equal number of rows for each class."""
    df_dep = pd.DataFrame({"text": ["depressed post"] * 100, "label": [1] * 100})
    df_happy = pd.DataFrame({"text": ["happy post"] * 50, "label": [0] * 50})

    balanced_df = balance_classes.balance_classes(df_dep, df_happy, target_per_class=50, random_state=42)

    assert isinstance(balanced_df, pd.DataFrame)
    assert balanced_df["label"].value_counts().to_dict() == {1: 50, 0: 50}
