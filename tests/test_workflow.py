from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

import prepro


def test_workflow_non_interactive():
    # Create test dataframe
    data = {
        "id_col": [1, 2, 3, 4, 5, 5],
        "const_col": ["a", "a", "a", "a", "a", "a"],
        "str_col": [" Apple  ", "banana", "apple", "APPLE", "banana", "banana"],
        "num_col": [1.0, 2.0, np.nan, 100.0, 2.0, 2.0],
        "date_col": [
            "2026-06-11",
            "2026-06-12",
            "2026-06-13",
            "2026-06-14",
            "2026-06-15",
            "2026-06-15",
        ],
        "target": [0, 1, 0, 1, 0, 0],
    }
    df = pd.DataFrame(data)

    # Run workflow with UI=False and split=False
    res = prepro.workflow(
        df,
        target_col="target",
        UI=False,
        run_target=True,
        run_duplicates=True,
        run_drop_useless=True,
        run_clean_strings=True,
        run_cast=True,
        run_extract_datetime=True,
        run_missing=True,
        run_outliers=True,
        run_skewness=True,
        run_scale=True,
        run_encode=True,
        run_collinearity=True,
        run_variance_filter=True,
        run_balance=False,
        run_polynomial=False,
        run_split=False,
        report=True,
    )

    # Basic assertions
    assert isinstance(res, pd.DataFrame)
    # Check that duplicates were removed
    assert len(res) < len(df)
    # Check that Constant columns were dropped
    assert "const_col" not in res.columns
    # Check that target column remains
    assert "target" in res.columns


def test_workflow_interactive():
    data = {
        "id_col": [1, 2, 3, 4, 5, 5],
        "const_col": ["a", "a", "a", "a", "a", "a"],
        "str_col": [" Apple  ", "banana", "apple", "APPLE", "banana", "banana"],
        "num_col": [1.0, 2.0, np.nan, 100.0, 2.0, 2.0],
        "date_col": [
            "2026-06-11",
            "2026-06-12",
            "2026-06-13",
            "2026-06-14",
            "2026-06-15",
            "2026-06-15",
        ],
        "target": [0, 1, 0, 1, 0, 0],
    }
    df = pd.DataFrame(data)

    # Mock inputs for the interactive wizard
    inputs = [
        "target",  # target column
        "y",
        "",
        "first",  # duplicates, subset (all), keep first
        "y",
        "0.9",
        "0.95",  # drop useless, id thresh, const thresh
        "y",
        "",
        "y",
        "y",
        "n",  # clean strings, cols (all), strip, lower, typos (no)
        "y",
        "n",
        "",  # cast, manual (no), na_strings (none)
        "y",
        "",
        "",
        "n",  # datetime, cols (all), features (all), cyclical (no)
        "y",
        "mean",
        "y",
        "y",
        "0.5",  # missing, mean, mcar, indicator, threshold
        "y",
        "iqr",
        "winsorize",
        "1.5",
        "",  # outliers, iqr, winsorize, threshold, cols
        "y",
        "yeojohnson",
        "0.5",
        "",  # skewness, yeojohnson, threshold, cols
        "y",
        "standard",
        "",  # scale, standard, cols
        "y",
        "auto",
        "y",  # encode, auto, drop_first
        "y",
        "both",
        "5.0",
        "0.9",
        "drop",  # collinearity, both, VIF, correlation, drop
        "y",
        "0.01",
        "0.95",  # variance filter, threshold, quasi threshold
        "n",  # class balance (no)
        "n",  # polynomial (no)
        "n",  # split dataset (no)
    ]

    with patch("builtins.input", side_effect=inputs):
        res = prepro.workflow(df, UI=True, report=True)

    assert isinstance(res, pd.DataFrame)
    assert len(res) < len(df)
    assert "const_col" not in res.columns
    assert "target" in res.columns


def test_workflow_non_interactive_split():
    # Test dataset splitting
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)

    res = prepro.workflow(
        df,
        target_col="target",
        UI=False,
        run_duplicates=False,
        run_drop_useless=False,
        run_clean_strings=False,
        run_cast=False,
        run_extract_datetime=False,
        run_missing=False,
        run_outliers=False,
        run_skewness=False,
        run_scale=False,
        run_encode=False,
        run_collinearity=False,
        run_variance_filter=False,
        run_split=True,
        split_params={"train_proportion": 0.8, "seed": 42},
    )

    assert isinstance(res, tuple)
    assert len(res) == 2
    train_df, test_df = res
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert len(train_df) == 8
    assert len(test_df) == 2
