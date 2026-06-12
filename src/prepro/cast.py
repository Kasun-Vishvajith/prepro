from typing import Dict, List, Optional

import pandas as pd


def cast(
    df: pd.DataFrame,
    cast_map: Optional[Dict[str, str]] = None,
    na_strings: Optional[List[str]] = None,
    report: bool = False
) -> pd.DataFrame:
    """
    Casts DataFrame columns to specified types or automatically infers types.
    Optionally replaces custom missing value strings (e.g. "?", "N/A") with pd.NA
    and prints a detailed report of the casting operations.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to process.
    cast_map : Dict[str, str], optional
        A dictionary mapping column names to target types:
        - "int" / "integer" -> Int64 (nullable integer)
        - "float" / "double" -> Float64 (nullable float)
        - "str" / "string" -> string
        - "bool" / "boolean" -> boolean (nullable boolean)
        - "datetime" / "date" -> datetime64[ns]
        - "category" -> category
        If None, the function will attempt automatic inference on object/string columns.
    na_strings : List[str], optional
        A list of string placeholders representing missing values to replace with pd.NA.
    report : bool, default False
        If True, prints a formatted report summarizing original types, new types,
        and missing values replaced.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with casted columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    # Create a copy to avoid in-place mutation of the original DataFrame
    df = df.copy()

    # Validate cast_map columns
    if cast_map is not None:
        for col in cast_map:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cast_map not found in DataFrame columns"
                )

    # Track replacements and original dtypes
    na_replaced_counts = {col: 0 for col in df.columns}
    original_dtypes = {col: str(df[col].dtype) for col in df.columns}

    # Replace custom NA strings
    if na_strings is not None and len(na_strings) > 0:
        na_strings_stripped = [s.strip() for s in na_strings]
        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)

            # Only scan string-like/object/categorical columns for custom NA strings
            if "str" in dtype_str or "object" in dtype_str or "category" in dtype_str:
                non_null_mask = series.notnull()
                if non_null_mask.any():
                    is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
                    if is_categorical:
                        series_for_check = series.astype(object).astype(str).str.strip()
                    else:
                        series_for_check = series.astype(str).str.strip()

                    match_mask = (
                        non_null_mask & series_for_check.isin(na_strings_stripped)
                    )
                    count = int(match_mask.sum())
                    if count > 0:
                        if is_categorical:
                            new_series = series.astype(object)
                            new_series[match_mask] = pd.NA
                            df[col] = new_series.astype("category")
                        else:
                            df.loc[match_mask, col] = pd.NA
                        na_replaced_counts[col] = count

    # Cast columns
    if cast_map is not None:
        for col, target_type in cast_map.items():
            target_type_lower = str(target_type).lower().strip()

            if target_type_lower in ["int", "integer", "int64"]:
                df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")
            elif target_type_lower in ["float", "double", "float64"]:
                df[col] = pd.to_numeric(df[col], errors='raise').astype("Float64")
            elif target_type_lower in ["str", "string"]:
                df[col] = df[col].astype("string")
            elif target_type_lower in ["bool", "boolean"]:
                val = df[col]
                if "str" in str(val.dtype) or "object" in str(val.dtype):
                    val_str = val.astype(str).str.strip().str.lower()
                    true_mask = val_str.isin(["true", "t", "yes", "y", "1", "1.0"])
                    false_mask = val_str.isin(["false", "f", "no", "n", "0", "0.0"])
                    new_val = pd.Series(pd.NA, index=val.index, dtype="boolean")
                    new_val[val.notnull() & true_mask] = True
                    new_val[val.notnull() & false_mask] = False
                    df[col] = new_val
                else:
                    df[col] = df[col].astype("boolean")
            elif target_type_lower in ["datetime", "date", "datetime64"]:
                df[col] = pd.to_datetime(df[col], errors='raise')
            elif target_type_lower in ["category", "categorical"]:
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(target_type)
    else:
        # Automatic type inference
        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)

            if "object" in dtype_str or "str" in dtype_str or "category" in dtype_str:
                # 1. Try to convert to numeric (int or float)
                try:
                    numeric_series = pd.to_numeric(series, errors='raise')
                    if pd.api.types.is_float_dtype(numeric_series):
                        non_null_vals = numeric_series.dropna()
                        if (
                            len(non_null_vals) > 0
                            and (non_null_vals == non_null_vals.round()).all()
                        ):
                            df[col] = numeric_series.astype("Int64")
                        else:
                            df[col] = numeric_series.astype("Float64")
                    else:
                        df[col] = numeric_series.astype("Int64")
                    continue
                except (ValueError, TypeError):
                    pass

                # 2. Try to convert to datetime
                non_null_strs = series.dropna().astype(str).str.strip()
                if len(non_null_strs) > 0:
                    date_pattern = (
                        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}'
                        r'(?::\d{2})?)?$'
                    )
                    date_pattern_alt = (
                        r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}(?:\s+\d{1,2}:\d{2}'
                        r'(?::\d{2})?)?$'
                    )
                    if (
                        non_null_strs.str.match(date_pattern).all()
                        or non_null_strs.str.match(date_pattern_alt).all()
                    ):
                        try:
                            df[col] = pd.to_datetime(series, errors='raise')
                            continue
                        except (ValueError, TypeError):
                            pass

                # 3. Try to convert to boolean
                if len(non_null_strs) > 0:
                    unique_vals = set(non_null_strs.str.lower().unique())
                    bool_set = {
                        "true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"
                    }
                    if unique_vals.issubset(bool_set):
                        true_mask = non_null_strs.str.lower().isin(
                            ["true", "t", "yes", "y", "1"]
                        )
                        false_mask = non_null_strs.str.lower().isin(
                            ["false", "f", "no", "n", "0"]
                        )
                        new_val = pd.Series(
                            pd.NA, index=series.index, dtype="boolean"
                        )
                        new_val[series.notnull() & true_mask] = True
                        new_val[series.notnull() & false_mask] = False
                        df[col] = new_val
                        continue

        # Use df.convert_dtypes() to convert remaining columns to modern nullable dtypes
        df = df.convert_dtypes()

    # Print report if requested
    if report:
        print("=" * 65)
        print("                        CASTING REPORT")
        print("=" * 65)
        headers = ["Column", "Original Dtype", "New Dtype", "NA Replaced"]
        rows = []
        for col in df.columns:
            rows.append([
                str(col),
                original_dtypes[col],
                str(df[col].dtype),
                str(na_replaced_counts.get(col, 0))
            ])

        # Calculate widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*headers))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            print(fmt.format(*row))
        print("=" * 65)

    return df
