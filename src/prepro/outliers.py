from typing import List, Optional

import numpy as np
import pandas as pd


def outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    treatment: str = "winsorize",
    threshold: float = 1.5,
    contamination: float = 0.05,
    cols: Optional[List[str]] = None,
    report: bool = False
) -> pd.DataFrame:
    """
    Detects and treats outliers in numeric columns of a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    method : {"iqr", "zscore", "modified_zscore", "isolation_forest", "lof"},
             default "iqr"
        Outlier detection method.
    treatment : {"remove", "winsorize", "flag"}, default "winsorize"
        - "remove" : Remove rows containing outliers in target columns.
        - "winsorize" : Cap outlier values at the boundary thresholds.
        - "flag" : Add binary indicator columns (e.g. 'col_outlier')
                   showing outlier status.
    threshold : float, default 1.5
        Threshold multiplier (IQR multiplier or Z-score cutoff).
    contamination : float, default 0.05
        The proportion of outliers expected in the dataset
        (for Isolation Forest and LOF).
    cols : list of str, optional
        Specific numeric columns to process. If None, uses all numeric columns.
    report : bool, default False
        If True, prints a summary report of outliers detected and treated.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with outliers treated.
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

    method_lower = method.lower().strip()
    treatment_lower = treatment.lower().strip()

    # Outlier statistics
    outlier_counts = {col: 0 for col in cols}
    total_rows = len(df)

    if method_lower in ["isolation_forest", "lof"]:
        # Multivariate methods: fit together on specified columns
        # Fill missing values with median for detection model
        X = df[cols].fillna(df[cols].median()).values
        if len(X) == 0:
            return df

        if method_lower == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(X)
        else:
            from sklearn.neighbors import LocalOutlierFactor
            clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            preds = clf.fit_predict(X)

        # -1 represents outliers, 1 represents inliers
        outlier_mask = pd.Series(preds == -1, index=df.index)
        outlier_count = int(outlier_mask.sum())

        for col in cols:
            # For reporting, attribute the joint outliers to each column
            outlier_counts[col] = outlier_count

        if treatment_lower == "remove":
            df = df.loc[~outlier_mask]
        elif treatment_lower == "flag":
            df["outlier_flag"] = outlier_mask.astype("Int64")
        elif treatment_lower == "winsorize":
            # For multivariate, we cap each column individually at IQR bounds
            for col in cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                if pd.api.types.is_integer_dtype(df[col]):
                    lower_is_int = float(lower).is_integer() if lower is not None else True
                    upper_is_int = float(upper).is_integer() if upper is not None else True
                    if not (lower_is_int and upper_is_int):
                        if isinstance(df[col].dtype, pd.api.extensions.ExtensionDtype):
                            df[col] = df[col].astype("Float64")
                        else:
                            df[col] = df[col].astype(float)
                df[col] = df[col].clip(lower=lower, upper=upper)

    else:
        # Univariate methods: process column-by-column
        union_outlier_mask = pd.Series(False, index=df.index)

        for col in cols:
            series = df[col]
            if series.isnull().all():
                continue

            # Calculate bounds
            if method_lower == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
            elif method_lower == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0 or np.isnan(std):
                    lower, upper = series.min(), series.max()
                else:
                    lower = mean - threshold * std
                    upper = mean + threshold * std
            elif method_lower == "modified_zscore":
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad == 0 or np.isnan(mad):
                    # Fallback to standard zscore
                    mean = series.mean()
                    std = series.std()
                    if std == 0 or np.isnan(std):
                        lower, upper = series.min(), series.max()
                    else:
                        lower = mean - threshold * std
                        upper = mean + threshold * std
                else:
                    lower = median - (threshold * mad / 0.6745)
                    upper = median + (threshold * mad / 0.6745)
            else:
                raise ValueError(f"Unknown outlier detection method: '{method}'")

            col_outlier_mask = (series < lower) | (series > upper)
            outlier_counts[col] = int(col_outlier_mask.sum())
            union_outlier_mask = union_outlier_mask | col_outlier_mask

            if treatment_lower == "winsorize":
                if pd.api.types.is_integer_dtype(df[col]):
                    lower_is_int = float(lower).is_integer() if lower is not None else True
                    upper_is_int = float(upper).is_integer() if upper is not None else True
                    if not (lower_is_int and upper_is_int):
                        if isinstance(df[col].dtype, pd.api.extensions.ExtensionDtype):
                            df[col] = df[col].astype("Float64")
                        else:
                            df[col] = df[col].astype(float)
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif treatment_lower == "flag":
                df[f"{col}_outlier"] = col_outlier_mask.astype("Int64")

        if treatment_lower == "remove":
            df = df.loc[~union_outlier_mask]

    if report:
        print("=" * 60)
        print("                 OUTLIER DETECTION & TREATMENT")
        print("=" * 60)
        print(f"Detection Method: {method_lower}")
        print(f"Treatment Action: {treatment_lower}")
        print("Outliers Detected Per Column:")
        for col in cols:
            count = outlier_counts[col]
            pct = (count / total_rows * 100) if total_rows > 0 else 0.0
            print(f"  - {col}: {count} outliers ({pct:.2f}%)")
        if treatment_lower == "remove":
            print(f"\nTotal rows removed: {total_rows - len(df)}")
            print(f"Rows remaining:     {len(df)}")
        elif treatment_lower == "flag":
            if method_lower in ["isolation_forest", "lof"]:
                print("\nAdded joint outlier flag column: 'outlier_flag'")
            else:
                print("\nAdded outlier indicator columns for each variable.")
        print("=" * 60)

    return df
