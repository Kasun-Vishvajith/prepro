from typing import List, Optional

import pandas as pd


def cardinality(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    high_threshold: int = 50,
    report: bool = False
) -> pd.DataFrame:
    """
    Analyzes the cardinality (number of unique values) of columns in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.
    cols : list of str, optional
        Specific columns to analyze. If None, analyzes all categorical
        (object, string, categorical) columns.
    high_threshold : int, default 50
        The threshold of unique values above which a column is classified as
        having 'High' cardinality.
    report : bool, default False
        If True, prints a formatted summary table to standard output.

    Returns:
    --------
    pd.DataFrame
        A summary DataFrame containing column names, unique counts, unique
        percentages, and cardinality status.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Automatically identify categorical columns if cols is None
    if cols is None:
        cols = [
            col for col in df.columns
            if "object" in str(df[col].dtype)
            or "str" in str(df[col].dtype)
            or "string" in str(df[col].dtype)
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ]
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    cardinality_data = []

    for col in cols:
        series = df[col]
        unique_count = int(series.nunique(dropna=True))
        non_null_count = int(series.count())
        unique_pct = (
            float(unique_count / non_null_count * 100)
            if non_null_count > 0
            else 0.0
        )

        status = "High" if unique_count > high_threshold else "Low"

        cardinality_data.append({
            "Column": col,
            "Unique Count": unique_count,
            "Unique Pct": round(unique_pct, 2),
            "Status": status
        })

    res_df = pd.DataFrame(cardinality_data)

    if res_df.empty:
        res_df = pd.DataFrame(
            columns=["Column", "Unique Count", "Unique Pct", "Status"]
        )

    if report:
        print("=" * 65)
        print("                      CARDINALITY REPORT")
        print("=" * 65)
        if not res_df.empty:
            headers = list(res_df.columns)
            rows = res_df.values.tolist()
            widths = [len(h) for h in headers]
            for row in rows:
                for i, val in enumerate(row):
                    widths[i] = max(widths[i], len(str(val)))

            fmt = "  ".join(f"{{:<{w}}}" for w in widths)
            print(fmt.format(*headers))
            print("-" * (sum(widths) + 2 * (len(widths) - 1)))
            for row in rows:
                print(fmt.format(*[str(val) for val in row]))
        else:
            print("No categorical columns analyzed.")
        print("=" * 65)

    return res_df
