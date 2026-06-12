from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd


def polynomial(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    degree: int = 2,
    interaction_only: bool = False,
    include_bias: bool = False,
    report: bool = False,
    fill_values: Optional[Dict[str, float]] = None,
    return_fill_values: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]:
    """
    Generates and appends polynomial and interaction features from numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    cols : list of str, optional
        Specific numeric columns to use. If None, uses all numeric columns.
    degree : int, default 2
        The polynomial degree.
    interaction_only : bool, default False
        If True, only interaction features (e.g. x1 * x2) are generated,
        excluding power features (e.g. x1^2).
    include_bias : bool, default False
        If True, includes a bias column (constant term of 1).
    report : bool, default False
        If True, prints a summary of the generated features.
    fill_values : dict, optional
        Pre-computed dictionary of column medians to fill NaNs before polynomial feature calculation.
    return_fill_values : bool, default False
        If True, returns the median dictionary along with the transformed DataFrame.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, dict)
        A new DataFrame with original columns and the new
        polynomial/interaction columns, and optionally the fill values dictionary.
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
        if return_fill_values:
            return df, {}
        return df

    from sklearn.preprocessing import PolynomialFeatures

    # Fill NaNs temporarily for feature calculation to avoid errors
    if fill_values is not None:
        X_poly = df[cols].fillna(fill_values)
        fill_values_calc = fill_values
    else:
        medians_series = df[cols].median()
        fill_values_calc = medians_series.to_dict()
        X_poly = df[cols].fillna(medians_series)

    # Generate features
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias,
    )
    poly_arr = poly.fit_transform(X_poly)
    feat_names = poly.get_feature_names_out(cols)

    poly_df = pd.DataFrame(poly_arr, columns=feat_names, index=df.index)

    # Filter and rename new columns (avoid duplicating original columns)
    new_col_mappings = {}
    new_cols_to_add = []

    for name in feat_names:
        if name == "1":
            new_col_mappings[name] = "bias"
            new_cols_to_add.append("bias")
        elif name in cols:
            # Skip original columns
            continue
        else:
            # Format and sanitize spaces (e.g. "col1 col2" -> "col1_x_col2")
            clean_name = name.replace(" ", "_x_")
            new_col_mappings[name] = clean_name
            new_cols_to_add.append(clean_name)

    poly_df = poly_df.rename(columns=new_col_mappings)

    # Append new columns to DataFrame
    df = pd.concat([df, poly_df[new_cols_to_add]], axis=1)

    if report:
        print("=" * 60)
        print("                 POLYNOMIAL FEATURES REPORT")
        print("=" * 60)
        print(f"Polynomial Degree:          {degree}")
        print(f"Interaction-Only Features:  {'Yes' if interaction_only else 'No'}")
        print(f"Input Numeric Columns:      {', '.join(cols)}")
        print(f"New Features Generated:     {len(new_cols_to_add)}")
        if new_cols_to_add:
            print("\nGenerated Features List:")
            print("  - " + "\n  - ".join(new_cols_to_add[:15]))
            if len(new_cols_to_add) > 15:
                print(f"  - ... and {len(new_cols_to_add) - 15} more.")
        print("=" * 60)

    if return_fill_values:
        return df, fill_values_calc
    return df
