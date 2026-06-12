import numpy as np
import pandas as pd

import prepro


def test_duplicates(capsys):
    data = {
        "A": [1, 2, 2, 3],
        "B": ["x", "y", "y", "z"]
    }
    df = pd.DataFrame(data)

    res = prepro.duplicates(df, subset=["A"], keep="first", report=True)
    assert len(res) == 3
    assert list(res["A"]) == [1, 2, 3]

    captured = capsys.readouterr()
    assert "DUPLICATE REPORT" in captured.out


def test_drop_useless(capsys):
    data = {
        "id_col": [1, 2, 3, 4, 5],
        "const_col": ["a", "a", "a", "a", "a"],
        "normal_col": [10, 20, 10, 20, 10]
    }
    df = pd.DataFrame(data)

    res = prepro.drop_useless(
        df, id_threshold=1.0, const_threshold=1.0, drop=True, report=True
    )
    assert "id_col" not in res.columns
    assert "const_col" not in res.columns
    assert "normal_col" in res.columns

    captured = capsys.readouterr()
    assert "USELESS COLUMNS REPORT" in captured.out


def test_clean_strings(capsys):
    data = {
        "A": [" Apple  ", "banana", "apple", "APPLE"],
        "B": ["cat", "dog", "doggy", "dog"]
    }
    df = pd.DataFrame(data)

    # test strip and lowercase
    res1 = prepro.clean_strings(df, cols=["A"], strip=True, lowercase=True)
    assert res1["A"].iloc[0] == "apple"
    assert res1["A"].iloc[3] == "apple"

    # test typo fixing (fuzzy merge doggy to dog)
    res2 = prepro.clean_strings(
        df, cols=["B"], strip=True, lowercase=True,
        fix_typos=True, typo_threshold=0.7, report=True
    )
    assert res2["B"].iloc[2] == "dog"

    captured = capsys.readouterr()
    assert "STRING CLEANING REPORT" in captured.out


def test_cardinality(capsys):
    data = {
        "A": ["a", "b", "c", "d"],
        "B": [1, 2, 3, 4]
    }
    df = pd.DataFrame(data)

    res = prepro.cardinality(df, high_threshold=2, report=True)
    assert len(res) == 1
    assert res["Column"].iloc[0] == "A"
    assert res["Status"].iloc[0] == "High"

    captured = capsys.readouterr()
    assert "CARDINALITY REPORT" in captured.out


def test_detect_types(capsys):
    data = {
        "col_num": [1, 2, 3, 4],
        "col_bool": ["t", "f", "t", "f"],
        "col_date": ["2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14"]
    }
    df = pd.DataFrame(data)

    res = prepro.detect_types(df, report=True)
    assert res["col_num"] == "numeric"
    assert res["col_bool"] == "boolean"
    assert res["col_date"] == "datetime"

    captured = capsys.readouterr()
    assert "FEATURE TYPE DETECTION" in captured.out


def test_extract_datetime(capsys):
    data = {
        "col_date": ["2026-06-11 12:00:00", "2026-06-12 13:00:00"]
    }
    df = pd.DataFrame(data)

    res = prepro.extract_datetime(
        df, features=["year", "month", "hour"], cyclical=True, report=True
    )
    assert "col_date_year" in res.columns
    assert "col_date_month" in res.columns
    assert "col_date_hour_sin" in res.columns
    assert "col_date_hour_cos" in res.columns

    captured = capsys.readouterr()
    assert "DATETIME EXTRACTION REPORT" in captured.out


def test_missing(capsys):
    data = {
        "col_num": [1.0, 2.0, np.nan, 4.0],
        "col_cat": ["a", "b", np.nan, "a"],
        "col_drop": [1.0, np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(data)

    res = prepro.missing(
        df,
        strategy="mean",
        mcar_test=True,
        add_indicator=True,
        threshold=0.5,
        report=True
    )

    # check col_drop is dropped
    assert "col_drop" not in res.columns
    # check imputation
    assert res["col_num"].isnull().sum() == 0
    assert res["col_cat"].isnull().sum() == 0
    # check indicator
    assert "col_num_nan" in res.columns

    captured = capsys.readouterr()
    assert "MISSING VALUE ANALYSIS & IMPUTATION" in captured.out


def test_outliers(capsys):
    data = {
        "col_num": [1, 2, 3, 100, 2, 1, 3]
    }
    df = pd.DataFrame(data)

    # Test winsorization
    res_win = prepro.outliers(
        df, method="iqr", treatment="winsorize", threshold=1.5, report=True
    )
    assert res_win["col_num"].max() < 100

    # Test remove
    res_rem = prepro.outliers(df, method="iqr", treatment="remove", threshold=1.5)
    assert len(res_rem) == 6

    captured = capsys.readouterr()
    assert "OUTLIER DETECTION & TREATMENT" in captured.out


def test_skewness(capsys):
    # Skewed data
    data = {
        "col_skew": [1.0, 2.0, 5.0, 10.0, 100.0, 2.0, 1.0]
    }
    df = pd.DataFrame(data)

    res = prepro.skewness(df, method="yeojohnson", skew_threshold=0.1, report=True)
    # transformed skewness should be closer to 0
    orig_skew = pd.Series(data["col_skew"]).skew()
    new_skew = res["col_skew"].skew()
    assert abs(new_skew) < abs(orig_skew)

    captured = capsys.readouterr()
    assert "SKEWNESS ANALYSIS REPORT" in captured.out


def test_scale(capsys):
    data = {
        "col_1": [1.0, 2.0, 3.0],
        "col_2": [10.0, 20.0, 30.0]
    }
    df = pd.DataFrame(data)

    res = prepro.scale(df, method="standard", report=True)
    assert np.allclose(res["col_1"].mean(), 0.0)
    assert np.allclose(res["col_1"].std(ddof=0), 1.0)

    captured = capsys.readouterr()
    assert "FEATURE SCALING REPORT" in captured.out


def test_encode(capsys):
    data = {
        "col_cat": ["apple", "banana", "apple", "banana"],
        "target": [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    # test binary encoding
    res_bin = prepro.encode(df, method="binary", report=True)
    assert "col_cat_bin_0" in res_bin.columns

    # test target encoding
    res_target = prepro.encode(df, method="target", target_col="target")
    assert res_target["col_cat"].iloc[0] == 0.0
    assert res_target["col_cat"].iloc[1] == 1.0

    captured = capsys.readouterr()
    assert "ENCODING REPORT" in captured.out


def test_balance(capsys):
    data = {
        "feature1": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5, 4.0],
        "feature2": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0],
        "target": [0, 0, 0, 0, 0, 1, 1]  # imbalanced but has 2 minority samples
    }
    df = pd.DataFrame(data)

    res = prepro.balance(
        df, target="target", method="smote", ratio=1.0, seed=42, report=True
    )
    assert res["target"].value_counts().loc[0] == res["target"].value_counts().loc[1]

    captured = capsys.readouterr()
    assert "CLASS IMBALANCE HANDLING REPORT" in captured.out


def test_variance_filter(capsys):
    data = {
        "low_var": [1.0001, 1.0, 1.0002, 1.0],
        "const": ["a", "a", "a", "b"],
        "normal": [1.0, 2.0, 3.0, 4.0]
    }
    df = pd.DataFrame(data)

    res = prepro.variance_filter(df, threshold=0.01, quasi_threshold=0.7, report=True)
    assert "low_var" not in res.columns
    assert "const" not in res.columns
    assert "normal" in res.columns

    captured = capsys.readouterr()
    assert "VARIANCE FILTER REPORT" in captured.out


def test_polynomial(capsys):
    data = {
        "A": [1.0, 2.0],
        "B": [10.0, 20.0]
    }
    df = pd.DataFrame(data)

    res = prepro.polynomial(
        df, degree=2, interaction_only=False, include_bias=False, report=True
    )
    assert "A_x_B" in res.columns
    assert "A^2" in res.columns

    captured = capsys.readouterr()
    assert "POLYNOMIAL FEATURES REPORT" in captured.out


def test_collinearity(capsys):
    x = np.array([1, 2, 3, 4, 5])
    data = {
        "A": x,
        "B": x * 2 + 0.001,  # highly collinear
        "C": [10, 20, 15, 30, 25]
    }
    df = pd.DataFrame(data)

    res = prepro.collinearity(
        df, method="both", vif_threshold=5.0, corr_threshold=0.9,
        treatment="drop", report=True
    )
    # One of A or B should be dropped
    assert ("A" not in res.columns) or ("B" not in res.columns)

    captured = capsys.readouterr()
    assert "COLLINEARITY REPORT" in captured.out
