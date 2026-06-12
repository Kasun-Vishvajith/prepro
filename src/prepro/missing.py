from typing import Optional, Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scipy.stats as stats


def littles_mcar_test(df: pd.DataFrame) -> dict:
    """
    Performs Little's MCAR (Missing Completely At Random) chi-square test.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return {"statistic": np.nan, "p_value": np.nan, "df": 0}

    # Drop columns with 100% missing or 0% missing
    cols_to_use = []
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if 0 < missing_count < len(df):
            cols_to_use.append(col)

    if not cols_to_use:
        return {"statistic": 0.0, "p_value": 1.0, "df": 0}

    df_num = df[cols_to_use].astype(float)
    n, k = df_num.shape

    # Use complete cases or mean-filled fallback to calculate grand covariance
    complete_cases = df_num.dropna()
    if len(complete_cases) < 5:
        grand_mean = df_num.mean()
        grand_cov = df_num.fillna(grand_mean).cov()
    else:
        grand_mean = complete_cases.mean()
        grand_cov = complete_cases.cov()

    # Avoid singular covariance matrix
    if np.linalg.cond(grand_cov.values) > 1e15:
        grand_cov = grand_cov + np.eye(k) * 1e-6

    # Get unique patterns of missingness
    missing_patterns = df_num.isnull().values
    unique_patterns, pattern_indices = np.unique(
        missing_patterns, axis=0, return_inverse=True
    )

    chi2 = 0.0
    dof = 0

    for i, pattern in enumerate(unique_patterns):
        indices = np.where(pattern_indices == i)[0]
        n_j = len(indices)

        # Ignore fully observed and fully missing patterns for the statistic
        if not np.any(pattern) or np.all(pattern):
            continue

        observed_cols = np.where(~pattern)[0]
        if len(observed_cols) == 0:
            continue

        # Subspace mean and covariance
        mean_j = df_num.iloc[indices, observed_cols].mean().values
        mean_grand_j = grand_mean.iloc[observed_cols].values

        cov_sub = grand_cov.iloc[observed_cols, observed_cols].values
        if np.linalg.cond(cov_sub) > 1e15:
            cov_sub = cov_sub + np.eye(len(observed_cols)) * 1e-6
        cov_sub_inv = np.linalg.pinv(cov_sub)

        diff = mean_j - mean_grand_j
        chi2 += n_j * diff.dot(cov_sub_inv).dot(diff)
        dof += len(observed_cols)

    dof = max(dof - k, 1)
    p_value = 1.0 - stats.chi2.cdf(chi2, dof)

    return {"statistic": chi2, "p_value": p_value, "df": dof}


def missing(
    df: pd.DataFrame,
    strategy: Optional[str] = None,
    knn_k: int = 5,
    mcar_test: bool = True,
    add_indicator: bool = False,
    threshold: float = 0.5,
    report: bool = False,
    fitted_state: Optional[Dict[str, Any]] = None,
    return_state: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Analyzes and imputes missing values in a pandas DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    strategy : str, optional
        Imputation strategy:
        - "mean" : Impute numeric with mean, categorical with mode.
        - "median" : Impute numeric with median, categorical with mode.
        - "mode" : Impute all columns with mode.
        - "knn" : Impute numeric columns with KNNImputer, categorical with mode.
        - "mice" : Impute numeric columns with MICE (IterativeImputer),
                   categorical with mode.
        - "indicator" : Do not impute, only add missing indicator columns
                        (same as strategy=None but with indicators).
        If None, performs analysis only.
    knn_k : int, default 5
        Number of neighbors for KNNImputer.
    mcar_test : bool, default True
        If True, runs Little's MCAR test.
    add_indicator : bool, default False
        If True, adds a binary column 'col_nan' for each column containing nulls.
    threshold : float, default 0.5
        Maximum ratio of missing values allowed. Columns with missing ratio
        above this threshold are dropped before imputation.
    report : bool, default False
        If True, prints missing values summary, Little's MCAR test p-value,
        and imputation action.
    fitted_state : Dict[str, Any], optional
        Pre-computed missing value statistics and models.
    return_state : bool, default False
        If True, returns the calculated/trained missing state.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, dict)
        A new DataFrame with missing values treated, and optionally the state.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()
    n_rows = len(df)

    if fitted_state is None:
        # 1. Analyze missingness per column
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / n_rows if n_rows > 0 else missing_counts * 0.0

        # 2. Drop columns exceeding threshold
        cols_to_drop = []
        for col in df.columns:
            if missing_ratios[col] > threshold:
                cols_to_drop.append(col)

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            # Recalculate counts/ratios for remaining columns
            missing_counts = df.isnull().sum()
            missing_ratios = missing_counts / n_rows if n_rows > 0 else missing_counts * 0.0

        # 3. Add indicators if requested
        indicator_cols_added = []
        if add_indicator or strategy == "indicator":
            for col in df.columns:
                if missing_counts[col] > 0:
                    ind_col_name = f"{col}_nan"
                    df[ind_col_name] = df[col].isnull().astype("Int64")
                    indicator_cols_added.append(ind_col_name)

        # 4. Perform imputation if a strategy is requested
        imputed_cols = []
        imputer_values = {}
        imputer_model = None
        strategy_lower = strategy.lower().strip() if strategy is not None else None
        numeric_cols = []
        categorical_cols = []

        if strategy_lower is not None and strategy_lower != "indicator":
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in indicator_cols_added
            ]
            categorical_cols = [
                c for c in df.columns
                if c not in numeric_cols and c not in indicator_cols_added
            ]

            # Handle numeric columns imputation
            numeric_cols_with_na = [c for c in numeric_cols if missing_counts[c] > 0]
            if numeric_cols_with_na:
                if strategy_lower in ["mean", "median", "mode"]:
                    for col in numeric_cols:
                        if strategy_lower == "mean":
                            val = df[col].mean()
                        elif strategy_lower == "median":
                            val = df[col].median()
                        else:
                            mode_val = df[col].mode()
                            val = mode_val.iloc[0] if not mode_val.empty else np.nan
                        imputer_values[col] = val

                        if col in numeric_cols_with_na:
                            mean_val = val
                            if pd.api.types.is_integer_dtype(df[col]) and not float(mean_val).is_integer():
                                if isinstance(df[col].dtype, pd.api.extensions.ExtensionDtype):
                                    df[col] = df[col].astype("Float64")
                                else:
                                    df[col] = df[col].astype(float)
                            df[col] = df[col].fillna(mean_val)
                            imputed_cols.append((col, strategy_lower))
                elif strategy_lower == "knn":
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=knn_k)
                    df_num = df[numeric_cols].astype(float)
                    imputed_arr = imputer.fit_transform(df_num)
                    imputer_model = imputer
                    for idx, col in enumerate(numeric_cols):
                        df[col] = imputed_arr[:, idx]
                        if col in numeric_cols_with_na:
                            imputed_cols.append((col, f"knn (k={knn_k})"))
                elif strategy_lower == "mice":
                    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                    from sklearn.impute import IterativeImputer
                    imputer = IterativeImputer(random_state=42)
                    df_num = df[numeric_cols].astype(float)
                    imputed_arr = imputer.fit_transform(df_num)
                    imputer_model = imputer
                    for idx, col in enumerate(numeric_cols):
                        df[col] = imputed_arr[:, idx]
                        if col in numeric_cols_with_na:
                            imputed_cols.append((col, "mice"))
                else:
                    raise ValueError(f"Unknown imputation strategy: '{strategy}'")

            # Handle categorical columns imputation (always fall back to mode)
            for col in categorical_cols:
                mode_val = df[col].mode()
                val = mode_val.iloc[0] if not mode_val.empty else None
                imputer_values[col] = val
                if col in df.columns and df[col].isnull().any() and val is not None:
                    df[col] = df[col].fillna(val)
                    imputed_cols.append((col, "mode (fallback)"))

        state = {
            "cols_to_drop": cols_to_drop,
            "indicator_cols_added": indicator_cols_added,
            "imputer_values": imputer_values,
            "imputer_model": imputer_model,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "strategy": strategy_lower
        }
    else:
        # Use pre-fitted state
        cols_to_drop = fitted_state.get("cols_to_drop", [])
        indicator_cols_added = fitted_state.get("indicator_cols_added", [])
        imputer_values = fitted_state.get("imputer_values", {})
        imputer_model = fitted_state.get("imputer_model", None)
        numeric_cols = fitted_state.get("numeric_cols", [])
        categorical_cols = fitted_state.get("categorical_cols", [])
        strategy_lower = fitted_state.get("strategy", None)

        # 1. Drop columns
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # 2. Add indicators
        for col in indicator_cols_added:
            orig_col = col[:-4]  # strip '_nan'
            if orig_col in df.columns:
                df[col] = df[orig_col].isnull().astype("Int64")
            else:
                df[col] = pd.Series(0, index=df.index, dtype="Int64")

        # 3. Perform Imputation
        imputed_cols = []
        if strategy_lower is not None:
            if strategy_lower in ["mean", "median", "mode"]:
                for col, val in imputer_values.items():
                    if col in df.columns and df[col].isnull().any():
                        if pd.api.types.is_integer_dtype(df[col]) and not float(val).is_integer():
                            if isinstance(df[col].dtype, pd.api.extensions.ExtensionDtype):
                                df[col] = df[col].astype("Float64")
                            else:
                                df[col] = df[col].astype(float)
                        df[col] = df[col].fillna(val)
                        imputed_cols.append((col, strategy_lower))
            elif strategy_lower in ["knn", "mice"]:
                present_num_cols = [c for c in numeric_cols if c in df.columns]
                if present_num_cols:
                    df_num = df[present_num_cols].astype(float)
                    if len(present_num_cols) == len(numeric_cols) and imputer_model is not None:
                        imputed_arr = imputer_model.transform(df_num)
                        for idx, col in enumerate(numeric_cols):
                            df[col] = imputed_arr[:, idx]
                            imputed_cols.append((col, strategy_lower))
                    else:
                        # Fallback if columns mismatch
                        for col in present_num_cols:
                            val = imputer_values.get(col, df[col].mean())
                            df[col] = df[col].fillna(val)
                            imputed_cols.append((col, f"{strategy_lower} (fallback)"))

            for col in categorical_cols:
                if col in df.columns and df[col].isnull().any() and col in imputer_values:
                    val = imputer_values[col]
                    if val is not None:
                        df[col] = df[col].fillna(val)
                        imputed_cols.append((col, "mode (fallback)"))

        state = fitted_state

    # 5. Run Little's MCAR test if requested (only in fit mode)
    mcar_results = None
    if mcar_test and fitted_state is None:
        try:
            mcar_results = littles_mcar_test(df)
        except Exception:
            pass

    # 6. Print report
    if report:
        print("=" * 65)
        print("                 MISSING VALUE ANALYSIS & IMPUTATION")
        print("=" * 65)
        if fitted_state is None:
            print("Missing Value Distribution:")
            has_missing = False
            for col in missing_counts.index:
                count = missing_counts[col]
                if count > 0:
                    ratio = missing_ratios[col]
                    print(f"  - {col}: {count} missing ({ratio * 100:.2f}%)")
                    has_missing = True
            if not has_missing:
                print("  No missing values detected.")

            if cols_to_drop:
                print(f"\nDropped columns (> {threshold * 100:.1f}% missing):")
                print(f"  {', '.join(cols_to_drop)}")

            if mcar_results and not np.isnan(mcar_results["p_value"]):
                print("\nLittle's MCAR Test Results:")
                print(f"  - Chi-Square Statistic: {mcar_results['statistic']:.4f}")
                print(f"  - Degrees of Freedom:   {mcar_results['df']}")
                print(f"  - p-value:              {mcar_results['p_value']:.4f}")
                p_val = mcar_results["p_value"]
                if p_val < 0.05:
                    print(
                        "  - Interpretation: Reject MCAR "
                        "(Missing Not Completely At Random)"
                    )
                else:
                    print(
                        "  - Interpretation: Fail to reject MCAR "
                        "(Missing Completely At Random)"
                    )

            if indicator_cols_added:
                print("\nAdded Missing Indicators:")
                print(f"  {', '.join(indicator_cols_added)}")
        else:
            print("Transforming using pre-fitted missing values state.")

        if imputed_cols:
            print(f"\nImputed Columns (strategy: {strategy}):")
            for col, strat in imputed_cols:
                print(f"  - {col}: imputed via '{strat}'")
        elif strategy is not None and strategy != "indicator":
            print(
                f"\nImputation requested (strategy: {strategy}) "
                "but no missing values were present."
            )

        print("=" * 65)

    if return_state:
        return df, state
    return df
