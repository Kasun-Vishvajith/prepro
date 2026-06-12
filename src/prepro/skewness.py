from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def compute_skew_and_normality(
    series: pd.Series, normality_test: str
) -> tuple:
    """
    Computes skewness and the normality test p-value for a pandas Series.
    """
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan, np.nan

    skew_val = float(vals.skew())
    p_val = np.nan

    if normality_test == "shapiro":
        if len(vals) >= 3:
            # Shapiro-Wilk is limited to <= 5000 samples
            sample_vals = vals.sample(min(5000, len(vals)), random_state=42)
            _, p_val = stats.shapiro(sample_vals)
    elif normality_test == "dagostino":
        if len(vals) >= 8:
            _, p_val = stats.normaltest(vals)

    return skew_val, p_val


def skewness(
    df: pd.DataFrame,
    method: Optional[str] = None,
    skew_threshold: float = 0.5,
    normality_test: str = "shapiro",
    cols: Optional[List[str]] = None,
    report: bool = False
) -> pd.DataFrame:
    """
    Analyzes skewness and applies mathematical transformations to correct skewness.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    method : {"log", "sqrt", "boxcox", "yeojohnson", "auto"}, optional
        Transformation method. If None, performs analysis only.
        - "log" : Applies log1p (with automatic shift if negative values exist).
        - "sqrt" : Applies square root (with automatic shift if negative values exist).
        - "boxcox" : Applies Box-Cox power transform (with automatic shift if <= 0).
        - "yeojohnson" : Applies Yeo-Johnson power transform.
        - "auto" : Automatically applies Yeo-Johnson if skewness exceeds threshold.
    skew_threshold : float, default 0.5
        Transformation is only applied if absolute skewness exceeds this threshold.
    normality_test : {"shapiro", "dagostino"}, default "shapiro"
        Statistical test used to evaluate normality (p-value).
    cols : list of str, optional
        Specific columns to analyze and transform. If None, uses all numeric columns.
    report : bool, default False
        If True, prints a summary table of skewness before and after transformations.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with transformed columns (if method is not None).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Automatically identify numeric columns if cols is None
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    if not cols:
        return df

    method_lower = method.lower().strip() if method is not None else None
    norm_test_lower = normality_test.lower().strip()

    report_data = []

    for col in cols:
        series = df[col]
        orig_skew, orig_p = compute_skew_and_normality(series, norm_test_lower)

        applied_transform = "None"
        new_skew, new_p = orig_skew, orig_p

        # Decide whether to apply transformation
        if method_lower is not None and abs(orig_skew) > skew_threshold:
            selected_method = method_lower
            if method_lower == "auto":
                selected_method = "yeojohnson"

            # Apply transformations safely
            if selected_method == "log":
                min_val = series.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1.0
                    df[col] = np.log1p(series + shift)
                    applied_transform = f"log1p (shift={shift})"
                else:
                    df[col] = np.log1p(series)
                    applied_transform = "log1p"

            elif selected_method == "sqrt":
                min_val = series.min()
                if min_val < 0:
                    shift = abs(min_val)
                    df[col] = np.sqrt(series + shift)
                    applied_transform = f"sqrt (shift={shift})"
                else:
                    df[col] = np.sqrt(series)
                    applied_transform = "sqrt"

            elif selected_method == "boxcox":
                min_val = series.min()
                if min_val <= 0:
                    shift = abs(min_val) + 1.0
                    shifted = series + shift
                    transformed, _ = stats.boxcox(shifted)
                    df[col] = transformed
                    applied_transform = f"boxcox (shift={shift})"
                else:
                    transformed, _ = stats.boxcox(series)
                    df[col] = transformed
                    applied_transform = "boxcox"

            elif selected_method == "yeojohnson":
                transformed, _ = stats.yeojohnson(series)
                df[col] = transformed
                applied_transform = "yeojohnson"

            else:
                raise ValueError(
                    f"Unknown skewness correction method: '{method}'"
                )

            # Recalculate skew/normality after transform
            new_skew, new_p = compute_skew_and_normality(df[col], norm_test_lower)

        report_data.append({
            "Column": col,
            "Orig Skew": orig_skew,
            "Orig Norm p-val": orig_p,
            "Transform": applied_transform,
            "New Skew": new_skew,
            "New Norm p-val": new_p
        })

    if report:
        print("=" * 75)
        print("                      SKEWNESS ANALYSIS REPORT")
        print("=" * 75)
        headers = [
            "Column", "Orig Skew", f"Orig p ({norm_test_lower})",
            "Transform", "New Skew", f"New p ({norm_test_lower})"
        ]
        widths = [len(h) for h in headers]
        rows = []
        for rd in report_data:
            orig_p_str = (
                f"{rd['Orig Norm p-val']:.4f}"
                if not np.isnan(rd["Orig Norm p-val"])
                else "NaN"
            )
            new_p_str = (
                f"{rd['New Norm p-val']:.4f}"
                if not np.isnan(rd["New Norm p-val"])
                else "NaN"
            )
            row = [
                rd["Column"],
                f"{rd['Orig Skew']:.4f}" if not np.isnan(rd["Orig Skew"]) else "NaN",
                orig_p_str,
                rd["Transform"],
                f"{rd['New Skew']:.4f}" if not np.isnan(rd["New Skew"]) else "NaN",
                new_p_str
            ]
            rows.append(row)
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            print(fmt.format(*row))
        print("=" * 75)

    return df
