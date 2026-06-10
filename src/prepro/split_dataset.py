from typing import Tuple

import pandas as pd


def split(
    df: pd.DataFrame, train_proportion: float, seed: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing partitions.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to split.
    train_proportion : float
        The proportion of data to assign to the training set (between 0.0 and 1.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training and testing DataFrames.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    if not isinstance(train_proportion, (int, float)):
        raise TypeError("train_proportion must be a numeric value")
    if not (0.0 < train_proportion < 1.0):
        raise ValueError(
            "train_proportion must be between 0.0 and 1.0 (exclusive)"
        )

    train_df = df.sample(frac=train_proportion, random_state=seed)
    test_df = df.drop(train_df.index)

    return train_df, test_df
