import pandas as pd


def target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Selects the target variable column and removes all rows where the target
    variable is missing (either standard nulls or common placeholders).

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to clean.
    target_col : str
        The name of the target variable column.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with missing target rows removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame columns"
        )

    series = df[target_col]

    # 1. Check for standard missing values (NaN/None)
    is_null = series.isnull()

    # 2. Check for common placeholder missing values
    common_placeholders = {"?", "N/A", "n/a", "NA", "na", "nan", "None", ""}
    is_placeholder = pd.Series(False, index=df.index)

    # Only scan string-like/object/categorical columns for placeholders
    dtype_str = str(series.dtype)
    if "str" in dtype_str or "object" in dtype_str or "category" in dtype_str:
        is_placeholder = series.astype(str).str.strip().isin(common_placeholders)

    # Combine masks
    missing_mask = is_null | is_placeholder
    missing_count = int(missing_mask.sum())

    # Drop rows
    cleaned_df = df.loc[~missing_mask].copy()

    # Print log message
    print(
        f"Target preprocessing: Removed {missing_count} rows with missing "
        f"target values in '{target_col}'."
    )

    return cleaned_df
