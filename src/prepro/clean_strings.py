import difflib
from typing import List, Optional

import pandas as pd


def clean_strings(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    strip: bool = True,
    lowercase: bool = True,
    fix_typos: bool = False,
    typo_threshold: float = 0.85,
    report: bool = False
) -> pd.DataFrame:
    """
    Cleans string/text columns by trimming whitespace, normalizing casing,
    and optionally fixing typos/rare value variances via fuzzy string matching.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to clean.
    cols : list of str, optional
        The specific columns to clean. If None, cleans all object/string columns.
    strip : bool, default True
        If True, strips leading and trailing whitespace.
    lowercase : bool, default True
        If True, normalizes casing to lowercase.
    fix_typos : bool, default False
        If True, merges rare string values with highly similar frequent ones.
    typo_threshold : float, default 0.85
        Fuzzy matching similarity score (0.0 to 1.0) above which to merge typos.
    report : bool, default False
        If True, prints a summary report of string cleaning operations.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with cleaned string columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")

    df = df.copy()

    # Automatically identify string/object columns if cols is None
    if cols is None:
        cols = [
            col for col in df.columns
            if "object" in str(df[col].dtype)
            or "str" in str(df[col].dtype)
            or "string" in str(df[col].dtype)
            or isinstance(df[col].dtype, pd.CategoricalDtype)
        ]
    else:
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' in cols not found in DataFrame columns"
                )

    # Track metrics for the report
    strip_counts = {col: 0 for col in cols}
    lowercase_counts = {col: 0 for col in cols}
    typo_merge_counts = {col: 0 for col in cols}
    typo_mappings_report = {col: [] for col in cols}

    for col in cols:
        series = df[col]
        non_null_mask = series.notnull()
        if not non_null_mask.any():
            continue

        is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
        if is_categorical:
            cleaned_series = series.astype(object)
        else:
            cleaned_series = series.copy()

        # Whitespace stripping
        if strip:
            original = cleaned_series.astype(str)
            stripped = original.str.strip()
            diff_mask = non_null_mask & (original != stripped)
            strip_counts[col] = int(diff_mask.sum())
            cleaned_series[non_null_mask] = stripped[non_null_mask]

        # Case normalization
        if lowercase:
            original = cleaned_series.astype(str)
            lowercased = original.str.lower()
            diff_mask = non_null_mask & (original != lowercased)
            lowercase_counts[col] = int(diff_mask.sum())
            cleaned_series[non_null_mask] = lowercased[non_null_mask]

        # Fuzzy typo correction
        if fix_typos:
            val_counts = cleaned_series.dropna().value_counts()
            if len(val_counts) > 1:
                total_non_null = val_counts.sum()
                cum_sum = val_counts.cumsum()

                frequent_vals = []
                rare_vals = []
                for val, count in val_counts.items():
                    if len(frequent_vals) == 0:
                        frequent_vals.append(val)
                    elif cum_sum[val] / total_non_null <= 0.90:
                        frequent_vals.append(val)
                    else:
                        rare_vals.append(val)

                if frequent_vals and rare_vals:
                    typo_map = {}
                    for rare in rare_vals:
                        best_match = None
                        best_score = 0.0
                        for freq in frequent_vals:
                            score = difflib.SequenceMatcher(
                                None, str(rare), str(freq)
                            ).ratio()
                            if score >= typo_threshold and score > best_score:
                                best_score = score
                                best_match = freq
                        if best_match is not None:
                            typo_map[rare] = best_match
                            typo_mappings_report[col].append(
                                (rare, best_match, best_score)
                            )

                    if typo_map:
                        affected_mask = cleaned_series.isin(typo_map.keys())
                        typo_merge_counts[col] = int(affected_mask.sum())
                        cleaned_series = cleaned_series.replace(typo_map)

        if is_categorical:
            df[col] = cleaned_series.astype("category")
        else:
            df[col] = cleaned_series

    if report:
        print("=" * 60)
        print("                 STRING CLEANING REPORT")
        print("=" * 60)
        for col in cols:
            print(f"Column: {col}")
            if strip:
                print(f"  - Whitespace trimmed: {strip_counts[col]} values")
            if lowercase:
                print(f"  - Normalized to lowercase: {lowercase_counts[col]} values")
            if fix_typos:
                print(f"  - Typo merges: {typo_merge_counts[col]} values")
                if typo_mappings_report[col]:
                    print("    Merges detail:")
                    for rare, freq, score in typo_mappings_report[col]:
                        print(
                            f"      '{rare}' -> '{freq}' "
                            f"(similarity: {score:.2f})"
                        )
            print()
        print("=" * 60)

    return df
