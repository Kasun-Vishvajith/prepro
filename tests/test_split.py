import pandas as pd
import pytest

from prepro.split_dataset import split


def test_split_proportion():
    # Create a DataFrame of 10 rows
    df = pd.DataFrame({"A": range(10)})

    # Split 80/20
    train, test = split(df, 0.8, seed=1234)

    assert len(train) == 8
    assert len(test) == 2
    assert len(train) + len(test) == 10

    # Ensure all original rows are accounted for without duplicates
    combined = pd.concat([train, test]).sort_index()
    pd.testing.assert_frame_equal(combined, df)

def test_split_seeding():
    df = pd.DataFrame({"A": range(100)})

    # Split with same seed should yield identical sets
    train1, test1 = split(df, 0.7, seed=42)
    train2, test2 = split(df, 0.7, seed=42)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)

def test_split_invalid_proportions():
    df = pd.DataFrame({"A": [1, 2]})

    with pytest.raises(ValueError):
        split(df, 1.5)

    with pytest.raises(ValueError):
        split(df, -0.1)

def test_split_invalid_types():
    with pytest.raises(TypeError):
        split([1, 2, 3], 0.8)

    with pytest.raises(TypeError):
        split(pd.DataFrame({"A": [1]}), "0.8")
