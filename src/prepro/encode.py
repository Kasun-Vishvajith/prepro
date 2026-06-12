from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def encode(
    df: pd.DataFrame,
    method: str = "auto",
    ordinal_map: Optional[Dict[str, List[Any]]] = None,
    target_col: Optional[str] = None,
    drop_first: bool = True,
    handle_unknown: str = "ignore",
    report: bool = False,
    fitted_state: Optional[Dict[str, Any]] = None,
    return_state: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Encodes categorical (object, category, string) columns to numeric values.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process.
    method : {"auto", "onehot", "ordinal", "target", "frequency", "binary"},
             default "auto"
        Encoding strategy:
        - "onehot" : One-Hot Encoding (pd.get_dummies).
        - "ordinal" : Maps categories to ordered integers. Uses ordinal_map if
                      provided, otherwise uses alphabetical rank.
        - "target" : Replaces categories with target variable mean
                     (requires target_col).
        - "frequency" : Replaces categories with their relative frequency.
        - "binary" : Converts categories to integers, then splits binary
                     representation into columns.
        - "auto" : Auto-selects method based on cardinality:
                   - <= 2 unique values -> Label encoding (0/1).
                   - <= 10 unique values -> One-Hot Encoding.
                   - > 10 unique values -> Target encoding (if target_col
                     available) otherwise Frequency encoding.
    ordinal_map : Dict[str, list], optional
        Dictionary mapping column names to lists of ordered categories.
    target_col : str, optional
        Name of the target column. Required for target encoding.
    drop_first : bool, default True
        If True, drops the first dummy column in One-Hot encoding to avoid collinearity.
    handle_unknown : {"ignore", "error"}, default "ignore"
        Handling of unknown categories in ordinal encoding:
        - "ignore" : Maps unknown values to pd.NA.
        - "error" : Raises ValueError.
    report : bool, default False
        If True, prints a summary of encoded columns and methods used.
    fitted_state : Dict[str, Any], optional
        Pre-computed category mappings and encoding models.
    return_state : bool, default False
        If True, returns the calculated encoding state.

    Returns:
    --------
    pd.DataFrame or (pd.DataFrame, dict)
        A new DataFrame with encoded columns, and optionally the state.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Identify categorical columns to encode
    if fitted_state is not None:
        cat_cols = list(fitted_state.keys())
    else:
        cat_cols = [
            col for col in df.columns
            if "object" in str(df[col].dtype)
            or "str" in str(df[col].dtype)
            or "string" in str(df[col].dtype)
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ]
        # Exclude target column from encoding if present
        if target_col in cat_cols:
            cat_cols.remove(target_col)

    if not cat_cols:
        if return_state:
            return df, {}
        return df

    method_lower = method.lower().strip()
    handle_unknown_lower = handle_unknown.lower().strip()

    report_details = []
    state = {}

    if fitted_state is None:
        for col in cat_cols:
            if col not in df.columns:
                continue
            series = df[col]
            n_unique = series.nunique(dropna=True)

            selected_method = method_lower
            if method_lower == "auto":
                if n_unique <= 2:
                    selected_method = "label"
                elif n_unique <= 10:
                    selected_method = "onehot"
                else:
                    selected_method = "target" if target_col is not None else "frequency"

            col_state = {"method": selected_method}

            if selected_method == "label":
                # Simple label encoding (map categories to 0 and 1, keep NA)
                unique_vals = sorted(series.dropna().unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                df[col] = series.map(mapping).astype("Int64")
                report_details.append((col, "label (binary mapping)", 1))
                col_state["mapping"] = mapping

            elif selected_method == "onehot":
                # One-hot encoding
                dummies = pd.get_dummies(
                    series, prefix=col, drop_first=drop_first, dummy_na=False, dtype=int
                )
                df = df.drop(columns=[col])
                # Insert new dummy columns where original was
                df = pd.concat([df, dummies], axis=1)
                report_details.append((col, "onehot", len(dummies.columns)))
                col_state["columns"] = list(dummies.columns)

            elif selected_method == "ordinal":
                # Ordinal mapping
                mapping = {}
                if ordinal_map is not None and col in ordinal_map:
                    ordered_list = ordinal_map[col]
                    mapping = {val: idx for idx, val in enumerate(ordered_list)}
                else:
                    # Default alphabetical sorting
                    unique_vals = sorted(series.dropna().unique())
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}

                # Validate unknown values if handle_unknown is "error"
                if handle_unknown_lower == "error":
                    for val in series.dropna().unique():
                        if val not in mapping:
                            raise ValueError(
                                f"Unknown category '{val}' encountered in column '{col}'"
                            )

                df[col] = series.map(mapping).astype("Int64")
                report_details.append((col, "ordinal", 1))
                col_state["mapping"] = mapping

            elif selected_method == "target":
                if target_col is None:
                    raise ValueError(
                        "target_col must be provided for target encoding."
                    )
                if target_col not in df.columns:
                    raise ValueError(
                        f"Target column '{target_col}' not found in DataFrame."
                    )

                # Target encoding: replace category with mean of target col
                target_means = df.groupby(col)[target_col].mean().to_dict()
                overall_mean = df[target_col].mean()
                df[col] = series.map(target_means).fillna(overall_mean)
                report_details.append((col, "target", 1))
                col_state["target_means"] = target_means
                col_state["overall_mean"] = overall_mean

            elif selected_method == "frequency":
                freqs = series.value_counts(normalize=True).to_dict()
                df[col] = series.map(freqs).fillna(0.0)
                report_details.append((col, "frequency", 1))
                col_state["freqs"] = freqs

            elif selected_method == "binary":
                # Binary encoding: convert to integer categories, then split to binary bits
                unique_vals = series.dropna().unique()
                n_cats = len(unique_vals)
                n_bits = int(np.ceil(np.log2(n_cats))) if n_cats > 0 else 1
                cat_to_int = {cat: i for i, cat in enumerate(unique_vals)}

                new_bin_cols = []
                for bit in range(n_bits):
                    bit_col_name = f"{col}_bin_{bit}"
                    df[bit_col_name] = series.map(
                        lambda x: (cat_to_int[x] >> bit) & 1
                        if pd.notnull(x) and x in cat_to_int
                        else pd.NA
                    ).astype("Int64")
                    new_bin_cols.append(bit_col_name)

                df = df.drop(columns=[col])
                report_details.append((col, "binary", len(new_bin_cols)))
                col_state["cat_to_int"] = cat_to_int
                col_state["n_bits"] = n_bits

            else:
                raise ValueError(f"Unknown encoding method: '{method}'")

            state[col] = col_state
    else:
        # Use pre-fitted state
        for col, col_state in fitted_state.items():
            if col not in df.columns:
                continue

            series = df[col]
            selected_method = col_state["method"]

            if selected_method == "label":
                mapping = col_state["mapping"]
                df[col] = series.map(mapping).astype("Int64")
                report_details.append((col, "label (binary mapping)", 1))

            elif selected_method == "onehot":
                dummy_cols = col_state["columns"]
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False, dtype=int)
                df = df.drop(columns=[col])
                # Align dummy columns
                for d_col in dummy_cols:
                    if d_col not in dummies.columns:
                        dummies[d_col] = 0
                dummies = dummies[dummy_cols]
                df = pd.concat([df, dummies], axis=1)
                report_details.append((col, "onehot", len(dummy_cols)))

            elif selected_method == "ordinal":
                mapping = col_state["mapping"]
                if handle_unknown_lower == "error":
                    for val in series.dropna().unique():
                        if val not in mapping:
                            raise ValueError(
                                f"Unknown category '{val}' encountered in column '{col}'"
                            )
                df[col] = series.map(mapping).astype("Int64")
                report_details.append((col, "ordinal", 1))

            elif selected_method == "target":
                target_means = col_state["target_means"]
                overall_mean = col_state["overall_mean"]
                df[col] = series.map(target_means).fillna(overall_mean)
                report_details.append((col, "target", 1))

            elif selected_method == "frequency":
                freqs = col_state["freqs"]
                df[col] = series.map(freqs).fillna(0.0)
                report_details.append((col, "frequency", 1))

            elif selected_method == "binary":
                cat_to_int = col_state["cat_to_int"]
                n_bits = col_state["n_bits"]
                new_bin_cols = []
                for bit in range(n_bits):
                    bit_col_name = f"{col}_bin_{bit}"
                    df[bit_col_name] = series.map(
                        lambda x: (cat_to_int[x] >> bit) & 1
                        if pd.notnull(x) and x in cat_to_int
                        else pd.NA
                    ).astype("Int64")
                    new_bin_cols.append(bit_col_name)
                df = df.drop(columns=[col])
                report_details.append((col, "binary", len(new_bin_cols)))

        state = fitted_state

    if report:
        print("=" * 65)
        print("                        ENCODING REPORT")
        print("=" * 65)
        headers = ["Column", "Method Used", "Output Columns Generated"]
        widths = [len(h) for h in headers]
        rows = [[col, m, str(cnt)] for col, m, cnt in report_details]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            print(fmt.format(*row))
        print("=" * 65)

    if return_state:
        return df, state
    return df
