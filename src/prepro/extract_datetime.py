from typing import List, Optional

import numpy as np
import pandas as pd


def extract_datetime(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    cyclical: bool = True,
    drop_original: bool = True,
    report: bool = False
) -> pd.DataFrame:
    """
    Extracts tabular features from datetime columns and optionally encodes them
    as cyclical features (sine and cosine components).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    cols : list of str, optional
        Specific columns to extract datetime features from. If None, auto-detects
        all datetime-like columns.
    features : list of str, optional
        The specific fields to extract. Supported:
        - "year", "month", "day", "dayofweek", "hour", "is_weekend", "quarter".
        By default, extracts all fields.
    cyclical : bool, default True
        If True, adds sin and cos encoding columns for "month", "dayofweek",
        and "hour".
    drop_original : bool, default True
        If True, drops the original datetime columns from the DataFrame.
    report : bool, default False
        If True, prints a summary of features extracted and columns generated.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with extracted datetime features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Auto-detect datetime columns if cols is None
    if cols is None:
        from prepro.detect_types import detect_types
        dtypes = detect_types(df)
        cols = [col for col, t in dtypes.items() if t == "datetime"]
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    if features is None:
        features = [
            "year", "month", "day", "dayofweek", "hour", "is_weekend", "quarter"
        ]

    new_cols_generated = []

    for col in cols:
        # Safely convert to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_series = pd.to_datetime(df[col], errors='coerce')
        else:
            dt_series = df[col]

        for feat in features:
            feat_lower = feat.lower().strip()
            new_col_name = f"{col}_{feat_lower}"

            if feat_lower == "year":
                df[new_col_name] = dt_series.dt.year.astype("Int64")
            elif feat_lower == "month":
                df[new_col_name] = dt_series.dt.month.astype("Int64")
                if cyclical:
                    month_val = dt_series.dt.month
                    df[f"{new_col_name}_sin"] = np.sin(2 * np.pi * month_val / 12)
                    df[f"{new_col_name}_cos"] = np.cos(2 * np.pi * month_val / 12)
                    new_cols_generated.extend(
                        [f"{new_col_name}_sin", f"{new_col_name}_cos"]
                    )
            elif feat_lower == "day":
                df[new_col_name] = dt_series.dt.day.astype("Int64")
            elif feat_lower == "dayofweek":
                df[new_col_name] = dt_series.dt.dayofweek.astype("Int64")
                if cyclical:
                    dow_val = dt_series.dt.dayofweek
                    df[f"{new_col_name}_sin"] = np.sin(2 * np.pi * dow_val / 7)
                    df[f"{new_col_name}_cos"] = np.cos(2 * np.pi * dow_val / 7)
                    new_cols_generated.extend(
                        [f"{new_col_name}_sin", f"{new_col_name}_cos"]
                    )
            elif feat_lower == "hour":
                df[new_col_name] = dt_series.dt.hour.astype("Int64")
                if cyclical:
                    hour_val = dt_series.dt.hour
                    df[f"{new_col_name}_sin"] = np.sin(2 * np.pi * hour_val / 24)
                    df[f"{new_col_name}_cos"] = np.cos(2 * np.pi * hour_val / 24)
                    new_cols_generated.extend(
                        [f"{new_col_name}_sin", f"{new_col_name}_cos"]
                    )
            elif feat_lower == "is_weekend":
                weekend_val = dt_series.dt.dayofweek
                is_we = pd.Series(pd.NA, index=df.index, dtype="Int64")
                is_we[weekend_val.notnull()] = (
                    weekend_val[weekend_val.notnull()].isin([5, 6]).astype(int)
                )
                df[new_col_name] = is_we
            elif feat_lower == "quarter":
                df[new_col_name] = dt_series.dt.quarter.astype("Int64")
            else:
                raise ValueError(f"Unknown datetime feature: '{feat}'")

            new_cols_generated.append(new_col_name)

        if drop_original:
            df = df.drop(columns=[col])

    if report:
        print("=" * 60)
        print("                DATETIME EXTRACTION REPORT")
        print("=" * 60)
        processed_str = ", ".join(cols) if cols else "None"
        print(f"Datetime Columns Processed: {processed_str}")
        print(f"Features Extracted:         {', '.join(features)}")
        cyclical_str = "Enabled" if cyclical else "Disabled"
        print(f"Cyclical Sin/Cos Encoding:  {cyclical_str}")
        dropped_str = "Yes" if drop_original else "No"
        print(f"Original Columns Dropped:   {dropped_str}")
        print(f"New Columns Generated:      {len(new_cols_generated)}")
        if new_cols_generated:
            print("  - " + "\n  - ".join(new_cols_generated[:10]))
            if len(new_cols_generated) > 10:
                print(f"  - ... and {len(new_cols_generated) - 10} more.")
        print("=" * 60)

    return df
