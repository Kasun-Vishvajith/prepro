from typing import List, Optional, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd


def outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    treatment: str = "winsorize",
    threshold: float = 1.5,
    contamination: float = 0.05,
    cols: Optional[List[str]] = None,
    report: bool = False,
    fitted_state: Optional[Dict[str, Any]] = None,
    return_state: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
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
    fitted_state : Dict[str, Any], optional
        Pre-computed outlier bounds and fitted models.
    return_state : bool, default False
        If True, returns the calculated/trained outlier state.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, dict)
        A new DataFrame with outliers treated, and optionally the state.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Automatically identify numeric columns if cols is None
    if cols is None:
        if fitted_state is not None:
            cols = fitted_state["cols"]
        else:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    if not cols:
        if return_state:
            return df, {}
        return df

    method_lower = method.lower().strip()
    treatment_lower = treatment.lower().strip()

    # Outlier statistics
    outlier_counts = {col: 0 for col in cols}
    total_rows = len(df)

    if fitted_state is None:
        # Fit phase
        bounds = {}
        medians = df[cols].median().to_dict()
        clf = None

        if method_lower in ["isolation_forest", "lof"]:
            X = df[cols].fillna(medians).values
            if len(X) == 0:
                if return_state:
                    return df, {}
                return df

            if method_lower == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=contamination, random_state=42)
                preds = clf.fit_predict(X)
            else:
                from sklearn.neighbors import LocalOutlierFactor
                clf = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True)
                clf.fit(X)
                preds = clf.predict(X)

            # -1 represents outliers, 1 represents inliers
            outlier_mask = pd.Series(preds == -1, index=df.index)
            outlier_count = int(outlier_mask.sum())

            for col in cols:
                outlier_counts[col] = outlier_count

            if treatment_lower == "remove":
                df = df.loc[~outlier_mask]
            elif treatment_lower == "flag":
                df["outlier_flag"] = outlier_mask.astype("Int64")
            elif treatment_lower == "winsorize":
                for col in cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    bounds[col] = (lower, upper)
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
            # Univariate methods
            union_outlier_mask = pd.Series(False, index=df.index)

            for col in cols:
                series = df[col]
                if series.isnull().all():
                    bounds[col] = (np.nan, np.nan)
                    continue

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

                bounds[col] = (lower, upper)
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

        state = {
            "cols": cols,
            "method": method_lower,
            "treatment": treatment_lower,
            "bounds": bounds,
            "medians": medians,
            "clf": clf
        }
    else:
        # Transform phase using fitted_state
        cols = fitted_state["cols"]
        method_lower = fitted_state["method"]
        treatment_lower = fitted_state["treatment"]
        bounds = fitted_state["bounds"]
        medians = fitted_state["medians"]
        clf = fitted_state["clf"]

        if method_lower in ["isolation_forest", "lof"]:
            X = df[cols].fillna(medians).values
            if len(X) == 0:
                if return_state:
                    return df, fitted_state
                return df
            preds = clf.predict(X)
            outlier_mask = pd.Series(preds == -1, index=df.index)
            outlier_count = int(outlier_mask.sum())
            for col in cols:
                outlier_counts[col] = outlier_count

            if treatment_lower == "winsorize":
                for col in cols:
                    if col in bounds:
                        lower, upper = bounds[col]
                        if pd.api.types.is_integer_dtype(df[col]):
                            lower_is_int = float(lower).is_integer() if lower is not None else True
                            upper_is_int = float(upper).is_integer() if upper is not None else True
                            if not (lower_is_int and upper_is_int):
                                if isinstance(df[col].dtype, pd.api.extensions.ExtensionDtype):
                                    df[col] = df[col].astype("Float64")
                                else:
                                    df[col] = df[col].astype(float)
                        df[col] = df[col].clip(lower=lower, upper=upper)
            elif treatment_lower == "remove":
                df = df.loc[~outlier_mask]
            elif treatment_lower == "flag":
                df["outlier_flag"] = outlier_mask.astype("Int64")
        else:
            # Univariate methods
            union_outlier_mask = pd.Series(False, index=df.index)
            for col in cols:
                if col not in df.columns or col not in bounds:
                    continue
                lower, upper = bounds[col]
                series = df[col]
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

        state = fitted_state

    if report:
        print("=" * 60)
        print("                 OUTLIER DETECTION & TREATMENT")
        print("=" * 60)
        print(f"Detection Method: {method_lower}")
        print(f"Treatment Action: {treatment_lower}")
        if fitted_state is None:
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
        else:
            print("Outliers processed using pre-fitted outlier state.")
        print("=" * 60)

    if return_state:
        return df, state
    return df
