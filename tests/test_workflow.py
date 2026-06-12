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
        "y",
        "n",
        "",  # 1. cast, manual (no), na_strings (none)
        "y",
        "",
        "first",  # 2. duplicates, subset (all), keep first
        "y",
        "0.9",
        "0.95",  # 3. drop useless, id thresh, const thresh
        "target",  # 4. target column
        "n",  # 5. split dataset (no)
        "y",
        "",
        "y",
        "y",
        "n",  # 6. clean strings, cols (all), strip, lower, typos (no)
        "y",
        "",
        "50",  # 7. cardinality, cols (all), threshold (50)
        "y",  # 8. feature type detection
        "y",
        "",
        "",
        "n",  # 9. datetime, cols (all), features (all), cyclical (no)
        "y",
        "mean",
        "y",
        "y",
        "0.5",  # 10. missing, mean, mcar, indicator, threshold
        "y",
        "iqr",
        "winsorize",
        "1.5",
        "",  # 11. outliers, iqr, winsorize, threshold, cols
        "y",
        "yeojohnson",
        "0.5",
        "",  # 12. skewness, yeojohnson, threshold, cols
        "y",
        "standard",
        "",  # 13. scale, standard, cols
        "y",
        "auto",
        "y",  # 14. encode, auto, drop_first
        "n",  # 15. class balance (no)
        "y",
        "0.01",
        "0.95",  # 16. variance filter, threshold, quasi threshold
        "n",  # 17. polynomial (no)
        "y",
        "both",
        "5.0",
        "0.9",
        "drop",  # 18. collinearity, both, VIF, correlation, drop
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


def test_workflow_leakage_guard():
    # Construct a dataset designed to check scaling and encoding leakage
    data = {
        "num": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # Let's run workflow with split
    train_df, test_df = prepro.workflow(
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
        run_scale=True,
        scale_params={"method": "standard"},
        run_encode=True,
        encode_params={"method": "onehot"},
        run_collinearity=False,
        run_variance_filter=False,
        run_split=True,
        split_params={"train_proportion": 0.8, "seed": 42},
        report=False
    )

    # Re-calculate train set original values (by tracing which indices were sampled)
    train_indices = df.sample(frac=0.8, random_state=42).index
    test_indices = df.index.difference(train_indices)

    # Let's check scaling:
    # Train mean and std of "num" before scaling:
    train_num_orig = df.loc[train_indices, "num"]
    train_mean = train_num_orig.mean()
    train_std = train_num_orig.std(ddof=0) # StandardScaler std uses ddof=0

    # Test values scaled should be (test_orig - train_mean) / train_std
    test_num_orig = df.loc[test_indices, "num"]
    expected_test_scaled = (test_num_orig - train_mean) / train_std

    # Check scaled values in output
    assert np.allclose(test_df["num"], expected_test_scaled)
