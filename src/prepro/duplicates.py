from typing import List, Optional, Union

import pandas as pd


def duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: Union[str, bool] = "first",
    report: bool = False
) -> pd.DataFrame:
    """
    Detects and removes duplicate rows from a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to clean.
    subset : list of str, optional
        Only consider certain columns for identifying duplicates.
        By default, uses all columns.
    keep : {'first', 'last', False}, default 'first'
        - 'first' : Drop duplicates except for the first occurrence.
        - 'last' : Drop duplicates except for the last occurrence.
        - False : Drop all duplicates.
    report : bool, default False
        If True, prints a summary report of duplicates found and removed.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with duplicates removed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if subset is not None:
        for col in subset:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in subset not found in DataFrame columns"
                )

    if keep not in ["first", "last", False]:
        raise ValueError("Parameter 'keep' must be 'first', 'last', or False")

    df = df.copy()

    # Calculate duplicate statistics
    dup_mask = df.duplicated(subset=subset, keep=keep)
    dup_count = int(dup_mask.sum())
    total_rows = len(df)
    dup_pct = (
        float(dup_count / total_rows * 100) if total_rows > 0 else 0.0
    )

    cleaned_df = df.drop_duplicates(subset=subset, keep=keep)

    if report:
        print("=" * 50)
        print("                DUPLICATE REPORT")
        print("=" * 50)
        print(f"Total Rows:                {total_rows}")
        print(f"Duplicate Rows Detected:   {dup_count} ({dup_pct:.2f}%)")
        print(f"Rows Remaining:            {len(cleaned_df)}")
        if subset is not None:
            print(f"Subset Columns Used:       {', '.join(subset)}")
        else:
            print("Subset Columns Used:       All Columns")
        print(f"Duplicate Keep Strategy:   {keep}")
        print("=" * 50)

    return cleaned_df
