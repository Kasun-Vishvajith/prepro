from typing import Any, Dict, List, Optional

import pandas as pd


def detect_types(
    df: pd.DataFrame,
    override: Optional[Dict[str, str]] = None,
    ordinal_map: Optional[Dict[str, List[Any]]] = None,
    datetime_formats: Optional[List[str]] = None,
    report: bool = False
) -> Dict[str, str]:
    """
    Detects the semantic feature types of all columns in a pandas DataFrame.

    Feature Types:
    --------------
    - "numeric" : continuous or discrete numbers
    - "nominal" : unordered categorical values
    - "ordinal" : ordered categorical values
    - "datetime" : dates and timestamps
    - "boolean" : binary true/false indicators

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze.
    override : Dict[str, str], optional
        Custom type mapping to bypass detection for specific columns.
    ordinal_map : Dict[str, list], optional
        Mapping of column names to their ordered categories. Columns in this
        mapping are automatically classified as "ordinal".
    datetime_formats : List[str], optional
        A list of date formats to try when detecting datetime columns.
    report : bool, default False
        If True, prints a summary list of detected feature types.

    Returns:
    --------
    Dict[str, str]
        A dictionary mapping column names to detected feature types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()
    detected = {}

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)

        # 1. Check override
        if override is not None and col in override:
            detected[col] = override[col]
            continue

        # 2. Check ordinal map
        if ordinal_map is not None and col in ordinal_map:
            detected[col] = "ordinal"
            continue

        # 3. Check boolean
        is_bool = False
        if "bool" in dtype_str or "boolean" in dtype_str:
            is_bool = True
        else:
            non_null = series.dropna()
            if len(non_null) > 0:
                unique_vals = set(non_null.unique())
                bool_vals = {
                    True, False, 1.0, 0.0, 1, 0, "true", "false",
                    "t", "f", "yes", "no", "y", "n"
                }
                # Check if unique values are boolean-like and <= 2 classes
                if (
                    unique_vals.issubset(bool_vals)
                    and len(unique_vals) <= 2
                ):
                    is_bool = True

        if is_bool:
            detected[col] = "boolean"
            continue

        # 4. Check datetime
        is_datetime = False
        if "datetime" in dtype_str or "date" in dtype_str:
            is_datetime = True
        else:
            non_null_strs = series.dropna().astype(str).str.strip()
            if len(non_null_strs) > 0:
                if datetime_formats:
                    for fmt in datetime_formats:
                        try:
                            pd.to_datetime(series, format=fmt, errors='raise')
                            is_datetime = True
                            break
                        except (ValueError, TypeError):
                            pass

                if not is_datetime:
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
                            pd.to_datetime(series, errors='raise')
                            is_datetime = True
                        except (ValueError, TypeError):
                            pass

        if is_datetime:
            detected[col] = "datetime"
            continue

        # 5. Check numeric
        if (
            pd.api.types.is_numeric_dtype(series)
            and not isinstance(series.dtype, pd.CategoricalDtype)
        ):
            detected[col] = "numeric"
            continue

        # 6. Default to nominal
        detected[col] = "nominal"

    if report:
        print("=" * 50)
        print("              FEATURE TYPE DETECTION")
        print("=" * 50)
        for col, t in detected.items():
            print(f"  - {col:<20}: {t}")
        print("=" * 50)

    return detected
