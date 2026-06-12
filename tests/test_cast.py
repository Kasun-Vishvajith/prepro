import pandas as pd
import pytest

from prepro import cast


def test_cast_explicit_mapping():
    data = {
        "col_int": ["1", "2", None, "4"],
        "col_float": ["1.1", "2.2", "3.3", None],
        "col_str": [10, 20, 30, 40],
        "col_bool": ["True", "false", "yes", "no"],
        "col_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "col_cat": ["a", "b", "a", "b"]
    }
    df = pd.DataFrame(data)

    cast_map = {
        "col_int": "int",
        "col_float": "float",
        "col_str": "str",
        "col_bool": "bool",
        "col_date": "datetime",
        "col_cat": "category"
    }

    res = cast(df, cast_map=cast_map)

    # Check dtypes
    assert str(res["col_int"].dtype) == "Int64"
    assert str(res["col_float"].dtype) == "Float64"
    assert str(res["col_str"].dtype) == "string"
    assert str(res["col_bool"].dtype) == "boolean"
    assert "datetime64" in str(res["col_date"].dtype)
    assert str(res["col_cat"].dtype) == "category"

    # Check values
    assert res["col_int"].iloc[0] == 1
    assert pd.isna(res["col_int"].iloc[2])
    assert res["col_float"].iloc[0] == 1.1
    assert res["col_str"].iloc[0] == "10"
    assert res["col_bool"].iloc[0] == True  # noqa: E712
    assert res["col_bool"].iloc[1] == False  # noqa: E712
    assert res["col_bool"].iloc[2] == True  # noqa: E712
    assert res["col_bool"].iloc[3] == False  # noqa: E712
    assert res["col_date"].iloc[0] == pd.Timestamp("2026-01-01")


def test_cast_auto_inference():
    data = {
        "col_int": ["10", "20", "30", None],
        "col_float": ["1.5", "2.5", None, "4.5"],
        "col_bool": ["yes", "no", "yes", "no"],
        "col_date": ["2026/06/11", "2026/06/12", "2026/06/13", "2026/06/14"],
        "col_mixed": ["1", "abc", "2", "def"]
    }
    df = pd.DataFrame(data)

    res = cast(df)

    # Check inferred types
    assert str(res["col_int"].dtype) == "Int64"
    assert str(res["col_float"].dtype) == "Float64"
    assert str(res["col_bool"].dtype) == "boolean"
    assert "datetime64" in str(res["col_date"].dtype)
    # converted from object by convert_dtypes
    assert str(res["col_mixed"].dtype) == "string"


def test_cast_na_strings():
    data = {
        "col_int": ["1", "?", "3", "N/A"],
        "col_float": ["1.1", "none", "3.3", " "],
        "col_cat": ["a", "b", "N/A", "a"]
    }
    df = pd.DataFrame(data)
    df["col_cat"] = df["col_cat"].astype("category")

    cast_map = {
        "col_int": "int",
        "col_float": "float",
        "col_cat": "category"
    }

    # Custom na_strings replacement
    res = cast(
        df,
        cast_map=cast_map,
        na_strings=["?", "N/A", "none", " "],
        report=False
    )

    # Check if they cast correctly without error
    # since placeholders are replaced by pd.NA
    assert str(res["col_int"].dtype) == "Int64"
    assert str(res["col_float"].dtype) == "Float64"
    assert str(res["col_cat"].dtype) == "category"

    assert pd.isna(res["col_int"].iloc[1])
    assert pd.isna(res["col_int"].iloc[3])
    assert pd.isna(res["col_float"].iloc[1])
    assert pd.isna(res["col_float"].iloc[3])
    assert pd.isna(res["col_cat"].iloc[2])


def test_cast_report(capsys):
    data = {
        "col_int": ["1", "?", "3", "4"],
        "col_float": ["1.1", "2.2", "3.3", "4.4"],
    }
    df = pd.DataFrame(data)

    cast(
        df,
        cast_map={"col_int": "int", "col_float": "float"},
        na_strings=["?"],
        report=True
    )

    captured = capsys.readouterr()
    assert "CASTING REPORT" in captured.out
    assert "col_int" in captured.out
    assert "col_float" in captured.out
    assert "Original Dtype" in captured.out
    assert "New Dtype" in captured.out
    assert "NA Replaced" in captured.out
    # col_int has 1 NA replaced, col_float has 0
    assert "1" in captured.out
    assert "0" in captured.out


def test_cast_invalid_inputs():
    # Invalid DataFrame input
    with pytest.raises(TypeError):
        cast([1, 2, 3])

    df = pd.DataFrame({"A": [1, 2]})

    # Non-existent column in cast_map
    with pytest.raises(ValueError):
        cast(df, cast_map={"B": "int"})

    # Invalid cast target type that pandas astype cannot handle
    with pytest.raises((ValueError, TypeError)):
        cast(df, cast_map={"A": "invalid_type_name"})

    # Value conversion failures
    df_fail = pd.DataFrame({"A": ["1", "two", "3"]})
    with pytest.raises((ValueError, TypeError)):
        cast(df_fail, cast_map={"A": "int"})
