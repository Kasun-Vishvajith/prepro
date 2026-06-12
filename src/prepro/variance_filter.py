import pandas as pd


def variance_filter(
    df: pd.DataFrame,
    threshold: float = 0.01,
    quasi_threshold: float = 0.95,
    report: bool = False
) -> pd.DataFrame:
    """
    Removes low-variance numeric columns and quasi-constant columns from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter.
    threshold : float, default 0.01
        Variance threshold. Numeric columns with variance below this are removed.
    quasi_threshold : float, default 0.95
        Quasi-constant threshold. Columns where a single value dominates
        more than this ratio of all non-null values are removed.
    report : bool, default False
        If True, prints a summary report of analyzed and dropped columns.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with low-variance and quasi-constant columns removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    drop_reasons = {}

    for col in df.columns:
        series = df[col]
        non_null_count = series.count()

        if non_null_count == 0:
            drop_reasons[col] = "100% missing values"
            continue

        # 1. Quasi-constant check (for all columns)
        most_frequent_ratio = (
            series.value_counts(normalize=True, dropna=True).max()
        )
        if most_frequent_ratio > quasi_threshold:
            val_repr = str(series.value_counts().index[0])
            drop_reasons[col] = (
                f"quasi-constant (value '{val_repr}' covers "
                f"{most_frequent_ratio * 100:.2f}%)"
            )
            continue

        # 2. Low variance check (for numeric columns only)
        if pd.api.types.is_numeric_dtype(series) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            var_val = series.var(ddof=1)
            if pd.isnull(var_val):
                # If only 1 sample, variance is NaN
                drop_reasons[col] = "variance is NaN (too few values)"
            elif var_val < threshold:
                drop_reasons[col] = (
                    f"low variance (variance = {var_val:.6f} < {threshold})"
                )

    cols_to_drop = list(drop_reasons.keys())
    cleaned_df = df.drop(columns=cols_to_drop)

    if report:
        print("=" * 70)
        print("                        VARIANCE FILTER REPORT")
        print("=" * 70)
        print(f"Variance Threshold:       {threshold}")
        print(f"Quasi-Constant Threshold: {quasi_threshold}")
        print(f"Total Columns Analyzed:   {len(df.columns)}")
        print(f"Total Columns Dropped:    {len(cols_to_drop)}")
        if cols_to_drop:
            print("\nDropped Columns Details:")
            for col, reason in drop_reasons.items():
                print(f"  - {col:<20} : {reason}")
        else:
            print("\nNo columns were dropped.")
        print(f"\nRemaining Columns:        {len(cleaned_df.columns)}")
        print("=" * 70)

    return cleaned_df
