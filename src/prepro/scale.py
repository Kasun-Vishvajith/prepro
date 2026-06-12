from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd


def scale(
    df: pd.DataFrame,
    method: str = "standard",
    cols: Optional[List[str]] = None,
    report: bool = False,
    fitted_scaler: Optional[Any] = None,
    return_scaler: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """
    Scales and normalizes numeric columns using scikit-learn preprocessing.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    method : {"standard", "minmax", "robust", "maxabs"}, default "standard"
        Scaling method to use.
        - "standard" : StandardScaler (centers mean to 0, variance to 1)
        - "minmax" : MinMaxScaler (scales values to range [0, 1])
        - "robust" : RobustScaler (scales using median and IQR, robust to outliers)
        - "maxabs" : MaxAbsScaler (scales each feature by its maximum absolute value)
    cols : list of str, optional
        Specific columns to scale. If None, scales all numeric columns.
    report : bool, default False
        If True, prints a summary table showing stats before and after scaling.
    fitted_scaler : Any, optional
        Pre-fitted scikit-learn scaler object.
    return_scaler : bool, default False
        If True, returns the trained scaler object along with the scaled DataFrame.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, Any)
        A new DataFrame with scaled numeric columns, and optionally the scaler.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Automatically identify numeric columns if cols is None
    if cols is None:
        if fitted_scaler is not None:
            # Use columns that the pre-fitted scaler expects (if available) or numeric
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    if not cols:
        if return_scaler:
            return df, None
        return df

    method_lower = method.lower().strip()

    # Store statistics for the report
    stats_before = {}
    stats_after = {}
    for col in cols:
        stats_before[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }

    # Fit and transform columns
    if fitted_scaler is not None:
        scaler = fitted_scaler
        scaled_values = scaler.transform(df[cols].astype(float))
    else:
        # Initialize correct scaler
        if method_lower == "standard":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method_lower == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method_lower == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif method_lower == "maxabs":
            from sklearn.preprocessing import MaxAbsScaler
            scaler = MaxAbsScaler()
        else:
            raise ValueError(f"Unknown scaling method: '{method}'")

        scaled_values = scaler.fit_transform(df[cols].astype(float))

    df[cols] = scaled_values

    # Store statistics after scaling
    for col in cols:
        stats_after[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }

    if report:
        print("=" * 80)
        print(f"                      FEATURE SCALING REPORT ({method_lower})")
        print("=" * 80)
        headers = ["Column", "Before (Mean/Std/Min/Max)", "After (Mean/Std/Min/Max)"]
        widths = [len(h) for h in headers]
        rows = []
        for col in cols:
            sb = stats_before[col]
            sa = stats_after[col]
            before_str = (
                f"{sb['mean']:.2f} / {sb['std']:.2f} / "
                f"{sb['min']:.2f} / {sb['max']:.2f}"
            )
            after_str = (
                f"{sa['mean']:.2f} / {sa['std']:.2f} / "
                f"{sa['min']:.2f} / {sa['max']:.2f}"
            )
            row = [col, before_str, after_str]
            rows.append(row)
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            print(fmt.format(*row))
        print("=" * 80)

    if return_scaler:
        return df, scaler
    return df
