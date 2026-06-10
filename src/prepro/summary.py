from typing import Any, Dict, List

import pandas as pd


class SummaryResult:
    """
    Custom return object for prepro.summary(df).
    Supports tuple-like double indexing:
    - [0] returns the general summary dictionary
    - [1] returns a 2D list of column-wise metrics
    """
    def __init__(self, general: Dict[str, Any], column_details: List[List[Any]]):
        self.general = general
        self.column_details = column_details

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.general
        elif index == 1:
            return self.column_details
        else:
            msg = (
                "SummaryResult index out of range. "
                "Use 0 for general summary or 1 for detailed table."
            )
            raise IndexError(msg)

    def __repr__(self) -> str:
        # Custom representation to output a nice clean view when printed
        lines = []
        lines.append("=" * 50)
        lines.append("                 DATASET SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Total Rows:                   {self.general['nrows']}")
        lines.append(f"Total Columns:                {self.general['ncolumns']}")
        missing_count = self.general["missing_rows"]
        missing_pct = self.general["missing_rows_pct"]
        lines.append(
            f"Rows with any missing values: {missing_count} ({missing_pct:.2f}%)"
        )
        lines.append("=" * 50)
        lines.append("\nColumn Details:")

        # Simple text representation of the 2D table
        headers = self.column_details[0]
        rows = self.column_details[1:]

        # Calculate widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines.append(fmt.format(*headers))
        lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for row in rows:
            lines.append(fmt.format(*[str(val) for val in row]))

        return "\n".join(lines)


def summary(df: pd.DataFrame) -> SummaryResult:
    """
    Analyzes a pandas DataFrame and returns a SummaryResult object.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to analyze.

    Returns:
    --------
    SummaryResult
        An indexable object containing general and column-wise metrics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    nrows, ncolumns = df.shape

    # Calculate rows with any missing values
    missing_rows_mask = df.isnull().any(axis=1)
    missing_rows_count = int(missing_rows_mask.sum())
    missing_rows_pct = float((missing_rows_count / nrows) * 100) if nrows > 0 else 0.0

    general_summary = {
        "nrows": nrows,
        "ncolumns": ncolumns,
        "missing_rows": missing_rows_count,
        "missing_rows_pct": missing_rows_pct
    }

    # Placeholders commonly used for missing data
    common_placeholders = {"?", "N/A", "n/a", "NA", "na", "nan", "None", ""}

    # Create the column details starting with headers
    column_details = [
        [
            "Column",
            "Dtype",
            "Missing Count",
            "Missing Pct",
            "Unique Count",
            "Common Placeholders",
        ]
    ]

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)

        # Standard missing values
        null_count = int(series.isnull().sum())
        null_pct = float((null_count / nrows) * 100) if nrows > 0 else 0.0

        # Unique count
        unique_count = int(series.nunique())

        # Check for placeholder string missing values in object/string/category columns
        placeholders_found = {}
        if "str" in dtype or "object" in dtype or "category" in dtype:
            for val in series.dropna().unique():
                val_str = str(val).strip()
                if val_str in common_placeholders or val in common_placeholders:
                    count = int((series == val).sum())
                    placeholders_found[str(val)] = count

        if placeholders_found:
            placeholder_str = ", ".join(
                f"'{k}': {v}" for k, v in placeholders_found.items()
            )
        else:
            placeholder_str = "None"

        column_details.append([
            str(col),
            dtype,
            null_count,
            round(null_pct, 2),
            unique_count,
            placeholder_str
        ])

    return SummaryResult(general_summary, column_details)
