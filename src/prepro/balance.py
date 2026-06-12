from typing import Optional

import numpy as np
import pandas as pd


def balance(
    df: pd.DataFrame,
    target: str,
    method: str = "smote",
    ratio: float = 1.0,
    seed: Optional[int] = None,
    report: bool = False
) -> pd.DataFrame:
    """
    Handles class imbalance in classification datasets by oversampling,
    undersampling, or calculating sample weights.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    target : str
        The name of the target class column.
    method : {"smote", "adasyn", "undersample", "weights"}, default "smote"
        - "smote" : Synthetic Minority Over-sampling Technique.
        - "adasyn" : Adaptive Synthetic sampling.
        - "undersample" : Random Under-sampling.
        - "weights" : Adds a 'sample_weight' column containing class weights.
    ratio : float, default 1.0
        The desired ratio of minority class to majority class (for binary class).
    seed : int, optional
        Random seed for reproducibility.
    report : bool, default False
        If True, prints class distributions before and after balancing.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with resampled observations or added sample weights.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    df = df.copy()
    y = df[target]
    X = df.drop(columns=[target])

    method_lower = method.lower().strip()

    # Calculate distributions before resampling
    dist_before = y.value_counts(dropna=False)
    total_before = len(y)

    resampled_df = df

    if method_lower == "weights":
        class_counts = y.value_counts()
        total = len(y)
        n_classes = len(class_counts)
        if n_classes > 0:
            class_weights = {
                cls: total / (n_classes * count)
                for cls, count in class_counts.items()
            }
        else:
            class_weights = {}

        df["sample_weight"] = y.map(class_weights)
        resampled_df = df

    elif method_lower in ["smote", "adasyn", "undersample"]:
        # Validate data types for synthetic oversampling
        if method_lower in ["smote", "adasyn"]:
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    f"Method '{method}' requires all features to be numeric. "
                    f"Non-numeric columns found: {non_numeric}. "
                    "Please encode them first."
                )

            # Impute missing values temporarily to avoid model crash
            if X.isnull().any().any():
                X = X.fillna(X.median())

        # Configure Sampler
        if method_lower in ["smote", "adasyn"]:
            min_class_size = y.value_counts().min()
            if min_class_size < 2:
                raise ValueError(
                    f"Method '{method}' requires at least 2 samples "
                    f"in the minority class. Minority class size is {min_class_size}."
                )
            k_neigh = min(5, min_class_size - 1)

            if method_lower == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    sampling_strategy=ratio, k_neighbors=k_neigh, random_state=seed
                )
            else:
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(
                    sampling_strategy=ratio, n_neighbors=k_neigh, random_state=seed
                )
        else:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)

        # Resample
        X_res, y_res = sampler.fit_resample(X, y)

        # Re-assemble DataFrame
        resampled_df = pd.DataFrame(X_res, columns=X.columns)
        resampled_df[target] = y_res

    else:
        raise ValueError(f"Unknown class balancing method: '{method}'")

    if report:
        dist_after = resampled_df[target].value_counts(dropna=False)
        total_after = len(resampled_df)

        print("=" * 60)
        print("                 CLASS IMBALANCE HANDLING REPORT")
        print("=" * 60)
        print(f"Balancing Method: {method_lower}")
        print("\nClass Distribution BEFORE:")
        for val, count in dist_before.items():
            pct = (count / total_before * 100) if total_before > 0 else 0.0
            print(f"  - {val}: {count} ({pct:.2f}%)")

        print("\nClass Distribution AFTER:")
        for val, count in dist_after.items():
            pct = (count / total_after * 100) if total_after > 0 else 0.0
            print(f"  - {val}: {count} ({pct:.2f}%)")

        if method_lower == "weights":
            print("\nCalculated Sample Weights (added as 'sample_weight'):")
            for val, count in dist_before.items():
                w = (
                    class_weights.get(val, 0.0)
                    if 'class_weights' in locals()
                    else 0.0
                )
                print(f"  - {val}: weight = {w:.4f}")
        print("=" * 60)

    return resampled_df
