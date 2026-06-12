from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd


def compute_vif_dict(data_df: pd.DataFrame, columns: List[str]) -> dict:
    """
    Computes VIF for each column in 'columns' using scikit-learn LinearRegression.
    """
    vif_dict = {}
    if len(columns) <= 1:
        for col in columns:
            vif_dict[col] = 1.0
        return vif_dict

    from sklearn.linear_model import LinearRegression

    for col in columns:
        X_other = data_df[columns].drop(columns=[col])
        # Impute missing values for the model fitting
        X_other = X_other.fillna(X_other.median())
        y_target = data_df[col].fillna(data_df[col].median())

        if y_target.std() == 0:
            vif_dict[col] = float("inf")
            continue

        # Fit target on all other features
        reg = LinearRegression().fit(X_other, y_target)
        r2 = reg.score(X_other, y_target)

        if r2 >= 1.0:
            vif_dict[col] = float("inf")
        else:
            vif_dict[col] = 1.0 / (1.0 - r2)

    return vif_dict


def collinearity(
    df: pd.DataFrame,
    method: str = "vif",
    vif_threshold: float = 5.0,
    corr_threshold: float = 0.85,
    treatment: str = "warn",
    report: bool = False,
    cols_to_drop: Optional[List[str]] = None,
    return_cols_to_drop: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """
    Detects and treats multicollinearity using VIF and Pearson correlation.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.
    method : {"vif", "correlation", "both"}, default "vif"
        Collinearity detection method.
    vif_threshold : float, default 5.0
        VIF threshold above which a feature is flagged/dropped.
    corr_threshold : float, default 0.85
        Absolute correlation threshold above which pairs are flagged/dropped.
    treatment : {"warn", "drop", "report"}, default "warn"
        - "warn" : Prints console warnings for collinear features (if report=True).
        - "drop" : Iteratively drops collinear features.
        - "report" : Performs analysis only.
    report : bool, default False
        If True, prints a detailed report of VIF and correlation issues.
    cols_to_drop : list of str, optional
        Pre-computed list of columns to drop to prevent leakage.
    return_cols_to_drop : bool, default False
        If True, returns the list of dropped columns.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, list)
        A new DataFrame with collinear columns treated, and optionally the dropped columns list.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    if cols_to_drop is not None:
        cleaned_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        if report:
            print("=" * 65)
            print("                      COLLINEARITY REPORT")
            print("=" * 65)
            print("Using pre-fitted collinearity state.")
            print(f"Dropped {len(cols_to_drop)} columns due to collinearity:")
            print(f"  {', '.join(cols_to_drop) if cols_to_drop else 'None'}")
            print("=" * 65)
        if return_cols_to_drop:
            return cleaned_df, cols_to_drop
        return cleaned_df

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        if return_cols_to_drop:
            return df, []
        return df

    method_lower = method.lower().strip()
    treatment_lower = treatment.lower().strip()

    # 1. Compute correlation issues
    corr_matrix = df[numeric_cols].fillna(df[numeric_cols].median()).corr().abs()
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_a = numeric_cols[i]
            col_b = numeric_cols[j]
            val = corr_matrix.loc[col_a, col_b]
            if val > corr_threshold:
                high_corr_pairs.append((col_a, col_b, val))

    # 2. Compute baseline VIFs
    baseline_vifs = {}
    if method_lower in ["vif", "both"]:
        try:
            baseline_vifs = compute_vif_dict(df, numeric_cols)
        except Exception:
            pass

    # 3. Handle Dropping
    dropped_cols = set()

    if treatment_lower == "drop":
        # First drop based on correlation (drop one of each pair)
        if method_lower in ["correlation", "both"]:
            for col_a, col_b, val in high_corr_pairs:
                if col_a in dropped_cols or col_b in dropped_cols:
                    continue
                # Drop the one with higher average correlation to all other variables
                mean_corr_a = corr_matrix[col_a].mean()
                mean_corr_b = corr_matrix[col_b].mean()
                if mean_corr_a > mean_corr_b:
                    dropped_cols.add(col_a)
                else:
                    dropped_cols.add(col_b)

        # Then iteratively drop based on VIF
        if method_lower in ["vif", "both"]:
            remaining_cols = [c for c in numeric_cols if c not in dropped_cols]
            while len(remaining_cols) > 1:
                try:
                    vifs = compute_vif_dict(df, remaining_cols)
                    max_col = None
                    max_vif = -1.0
                    for col, vif in vifs.items():
                        if vif > max_vif:
                            max_vif = vif
                            max_col = col

                    if max_vif > vif_threshold:
                        dropped_cols.add(max_col)
                        remaining_cols.remove(max_col)
                    else:
                        break
                except Exception:
                    break

        df = df.drop(columns=list(dropped_cols))

    # 4. Handle Reports / Warnings
    if report:
        print("=" * 65)
        print("                      COLLINEARITY REPORT")
        print("=" * 65)
        print(f"Detection Method:    {method_lower}")
        print(f"Treatment Action:    {treatment_lower}")

        if method_lower in ["correlation", "both"]:
            print(f"\nHighly Correlated Pairs (> {corr_threshold}):")
            if high_corr_pairs:
                for col_a, col_b, val in high_corr_pairs:
                    print(f"  - {col_a} & {col_b} : correlation = {val:.4f}")
            else:
                print("  None")

        if method_lower in ["vif", "both"] and baseline_vifs:
            print(f"\nVariance Inflation Factors (threshold = {vif_threshold}):")
            for col, vif in baseline_vifs.items():
                vif_str = f"{vif:.4f}" if not np.isinf(vif) else "inf"
                flag_str = " [EXCEEDS THRESHOLD]" if vif > vif_threshold else ""
                print(f"  - {col:<20} : VIF = {vif_str}{flag_str}")

        if treatment_lower == "drop":
            print(f"\nDropped {len(dropped_cols)} columns due to collinearity:")
            print(f"  {', '.join(dropped_cols) if dropped_cols else 'None'}")
        elif treatment_lower == "warn":
            # List warning columns
            warned = []
            if method_lower in ["correlation", "both"]:
                for col_a, col_b, val in high_corr_pairs:
                    warned.append(col_a)
                    warned.append(col_b)
            if method_lower in ["vif", "both"] and baseline_vifs:
                for col, vif in baseline_vifs.items():
                    if vif > vif_threshold:
                        warned.append(col)
            warned = sorted(list(set(warned)))
            if warned:
                print("\n[WARNING] Collinearity detected in columns:")
                print(f"  {', '.join(warned)}")
            else:
                print("\nNo collinearity warnings issued.")
        print("=" * 65)

    if return_cols_to_drop:
        return df, list(dropped_cols)
    return df
