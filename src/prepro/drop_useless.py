import pandas as pd


def drop_useless(
    df: pd.DataFrame,
    id_threshold: float = 1.0,
    const_threshold: float = 0.95,
    drop: bool = True,
    report: bool = False
) -> pd.DataFrame:
    """
    Identifies and removes ID-like (highly unique) and constant/near-constant columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect.
    id_threshold : float, default 1.0
        The minimum ratio of unique values to total rows to classify as ID-like.
    const_threshold : float, default 0.95
        The minimum ratio of the most frequent value to total non-null values
        to classify as constant/near-constant.
    drop : bool, default True
        If True, drops identified useless columns.
        If False, only reports them.
    report : bool, default False
        If True, prints a summary report of identified useless columns.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with useless columns dropped (if drop=True),
        otherwise the copied original DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()
    n_rows = len(df)

    id_cols = []
    const_cols = []

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_non_null = series.count()

        if n_rows > 0:
            unique_ratio = n_unique / n_rows

            if n_non_null > 0:
                most_frequent_ratio = (
                    series.value_counts(normalize=True, dropna=True).max()
                )
            else:
                most_frequent_ratio = 1.0
        else:
            unique_ratio = 0.0
            most_frequent_ratio = 1.0

        # Check if constant/near-constant or completely null
        if most_frequent_ratio >= const_threshold or n_non_null == 0:
            const_cols.append((col, most_frequent_ratio))
        # Check if ID-like
        elif unique_ratio >= id_threshold:
            id_cols.append((col, unique_ratio))

    cols_to_drop = [c[0] for c in id_cols] + [c[0] for c in const_cols]

    if drop:
        cleaned_df = df.drop(columns=cols_to_drop)
    else:
        cleaned_df = df

    if report:
        print("=" * 60)
        print("                 USELESS COLUMNS REPORT")
        print("=" * 60)
        print("Constant / Near-Constant Columns:")
        if const_cols:
            for col, ratio in const_cols:
                print(f"  - {col}: max value ratio = {ratio:.4f}")
        else:
            print("  None")

        print("\nID-like Columns:")
        if id_cols:
            for col, ratio in id_cols:
                print(f"  - {col}: unique ratio = {ratio:.4f}")
        else:
            print("  None")

        print("\nAction Taken:")
        if drop:
            dropped_str = (
                ", ".join(cols_to_drop) if cols_to_drop else "None"
            )
            print(f"  Dropped {len(cols_to_drop)} columns: {dropped_str}")
            print(f"  Columns remaining: {len(cleaned_df.columns)}")
        else:
            print("  None (drop=False, report only)")
        print("=" * 60)

    return cleaned_df
