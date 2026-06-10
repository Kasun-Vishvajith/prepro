import pandas as pd
import pytest

from prepro.target_preprocess import target


def test_target_cleaning(capsys):
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "target_var": [10.0, None, 30.0, "?", 50.0]
    }
    df = pd.DataFrame(data)

    cleaned_df = target(df, "target_var")

    # Verify rows with None and "?" were removed (indices 1 and 3)
    assert len(cleaned_df) == 3
    assert list(cleaned_df["feature1"]) == [1, 3, 5]
    assert list(cleaned_df["target_var"]) == [10.0, 30.0, 50.0]

    # Verify console output message
    captured = capsys.readouterr()
    assert "Removed 2 rows with missing target values" in captured.out

def test_target_missing_column():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(ValueError):
        target(df, "B")

def test_target_invalid_type():
    with pytest.raises(TypeError):
        target([1, 2, 3], "target_var")
