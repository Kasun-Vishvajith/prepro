import re
from typing import Union, List
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """
    Clean text input by removing extra whitespaces and lowercasing.

    Parameters:
    -----------
    text : str
        The input text to clean.

    Returns:
    --------
    str
        The cleaned text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    # Remove extra spaces and strip
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned.lower()

def _downcast_series(s: pd.Series, target_dtype: str) -> pd.Series:
    """
    Helper to downcast a pandas Series to save memory where safe.
    """
    dtype_name = s.dtype.name
    
    # 1. Nullable integer types
    if dtype_name in ["Int64", "Int32", "Int16", "Int8"]:
        non_na = s.dropna()
        if not non_na.empty:
            min_val = non_na.min()
            max_val = non_na.max()
            if min_val >= -128 and max_val <= 127:
                return s.astype("Int8")
            elif min_val >= -32768 and max_val <= 32767:
                return s.astype("Int16")
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return s.astype("Int32")
        else:
            return s.astype("Int8")
        return s
        
    # 2. Standard numpy integer types
    elif dtype_name in ["int64", "int32", "int16", "int8"]:
        min_val = s.min()
        max_val = s.max()
        if min_val >= -128 and max_val <= 127:
            return s.astype("int8")
        elif min_val >= -32768 and max_val <= 32767:
            return s.astype("int16")
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return s.astype("int32")
        return s
        
    # 3. Nullable float types
    elif dtype_name == "Float64":
        non_na = s.dropna()
        if not non_na.empty:
            min_val = non_na.min()
            max_val = non_na.max()
            if min_val >= -3.4028235e38 and max_val <= 3.4028235e38:
                return s.astype("Float32")
        else:
            return s.astype("Float32")
        return s
        
    # 4. Standard numpy float types
    elif dtype_name == "float64":
        min_val = s.min()
        max_val = s.max()
        if min_val >= -3.4028235e38 and max_val <= 3.4028235e38:
            return s.astype("float32")
        return s
        
    return s

def cast(
    df: pd.DataFrame,
    dtypes: dict,
    na_values: list = None,
    downcast: bool = False
) -> pd.DataFrame:
    """
    Cast columns of a pandas DataFrame to correct dtypes and map domain-specific
    missing-value codes to NaN/None in one pass.

    Parameters:
    -----------
    df : pd.DataFrame
        Input pandas DataFrame to cast.
    dtypes : dict
        Mapping of column name -> target dtype string.
    na_values : list[str], optional
        Extra strings/codes to treat as NaN. Defaults to ["?", "N/A", "n/a", "none", "None", "-", ""].
    downcast : bool, default False
        If True, attempt memory-saving downcasting after cast (e.g. Float64 -> Float32).

    Returns:
    --------
    pd.DataFrame
        New DataFrame with corrected dtypes. Original is not mutated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    
    if na_values is None:
        na_values = ["?", "N/A", "n/a", "none", "None", "-", ""]
    
    # Pre-resolve numeric equivalents of na_values to match if types are already numbers
    resolved_na = list(na_values)
    for val in na_values:
        if isinstance(val, str):
            try:
                resolved_na.append(int(val))
            except ValueError:
                pass
            try:
                resolved_na.append(float(val))
            except ValueError:
                pass

    unique_na = []
    for v in resolved_na:
        if v not in unique_na:
            unique_na.append(v)

    df_copy = df.copy()

    for col, target_dtype in dtypes.items():
        if col not in df_copy.columns:
            continue

        s = df_copy[col].astype(object)
        
        # Strip string values to match cleaned whitespace
        s = s.map(lambda x: x.strip() if isinstance(x, str) else x)
        s = s.replace(unique_na, pd.NA)

        # Boolean custom parsing
        if str(target_dtype).lower() in ["boolean", "bool"]:
            def to_bool(val):
                if pd.isna(val) or val is pd.NA:
                    return pd.NA
                if isinstance(val, bool):
                    return val
                val_str = str(val).strip().lower()
                if val_str in ["1", "true", "t", "yes", "y"]:
                    return True
                if val_str in ["0", "false", "f", "no", "n"]:
                    return False
                return val
            
            s = s.map(to_bool).astype("boolean")
        else:
            s = s.astype(target_dtype)

        if downcast:
            s = _downcast_series(s, target_dtype)

        df_copy[col] = s

    return df_copy

def duplicate(
    df: pd.DataFrame,
    key: Union[str, List[str]] = None,
    keep: str = "first",
    flag_col: str = None,
    on_conflict: str = "drop"
) -> pd.DataFrame:
    """
    Detect and resolve exact duplicates and key-level near-duplicates in one call.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to deduplicate.
    key : str or list of str, optional
        Business key column(s) to detect near-duplicates. If None, only full-row
        exact duplicates are removed.
    keep : {"first", "last", "none", "all"}, default "first"
        Which row to retain when duplicates are found.
        - "first": Retains first duplicate copy.
        - "last": Retains last duplicate copy.
        - "none": All duplicate copies are removed.
        - "all": No duplicate copies are removed.
    flag_col : str, optional
        If set, adds a boolean column with this name marking near-duplicate rows
        instead of dropping them.
    on_conflict : {"drop", "flag", "raise"}, default "drop"
        What to do when near-duplicates are found (same key, differing values).
        "drop" applies the keep rule silently.
        "flag" requires flag_col to be set.
        "raise" throws a ValueError with the conflicting keys.

    Returns:
    --------
    pd.DataFrame
        Deduplicated copy; original not mutated. If flag_col is set,
        returned DataFrame retains all rows with the flag column appended.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if keep not in ["first", "last", "none", "all"]:
        raise ValueError("Parameter 'keep' must be 'first', 'last', 'none', or 'all'.")

    if on_conflict not in ["drop", "flag", "raise"]:
        raise ValueError("Parameter 'on_conflict' must be 'drop', 'flag', or 'raise'.")

    if on_conflict == "flag" and flag_col is None:
        raise ValueError("Parameter 'flag_col' must be provided when on_conflict='flag'.")

    df_copy = df.copy()

    # Case 1: If key is None, handle full-row exact duplicates
    if key is None:
        if flag_col is not None:
            # Mark duplicate rows that would be dropped
            if keep == "all":
                df_copy[flag_col] = df_copy.duplicated(keep=False)
            elif keep == "none":
                df_copy[flag_col] = df_copy.duplicated(keep=False)
            else:
                df_copy[flag_col] = df_copy.duplicated(keep=keep)
            return df_copy
        else:
            if keep == "all":
                return df_copy
            elif keep == "none":
                return df_copy.drop_duplicates(keep=False)
            else:
                return df_copy.drop_duplicates(keep=keep)

    # Convert key to a list of columns
    key_cols = [key] if isinstance(key, str) else list(key)

    # Verify that key columns exist in df
    for col in key_cols:
        if col not in df_copy.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # 1. Identify unique rows of the DataFrame considering all columns
    df_unique = df_copy.drop_duplicates()

    # Count how many unique rows exist for each key combination
    key_counts = df_unique.groupby(key_cols).size().reset_index(name="_unique_rows_count")

    # Merge this count back to the original DataFrame
    df_merged = df_copy.merge(key_counts, on=key_cols, how="left")

    # A conflict exists if a key has more than one unique row
    has_conflict = df_merged["_unique_rows_count"] > 1

    # 2. Apply on_conflict action
    if on_conflict == "raise":
        if has_conflict.any():
            conflicting_keys = df_merged[has_conflict][key_cols].drop_duplicates()
            raise ValueError(f"Near-duplicates found for key(s):\n{conflicting_keys}")
        
        # If no conflicts, drop duplicates of keys based on keep rule
        if keep == "all":
            return df_copy
        elif keep == "none":
            return df_copy.drop_duplicates(subset=key_cols, keep=False)
        else:
            return df_copy.drop_duplicates(subset=key_cols, keep=keep)

    elif on_conflict == "flag":
        df_copy[flag_col] = has_conflict.values
        return df_copy

    else:  # on_conflict == "drop"
        if keep == "all":
            return df_copy
        elif keep == "none":
            return df_copy.drop_duplicates(subset=key_cols, keep=False)
        else:
            return df_copy.drop_duplicates(subset=key_cols, keep=keep)
