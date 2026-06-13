import pytest
import pandas as pd
import prepro

def test_duplicate_requires_dataframe():
    with pytest.raises(TypeError):
        prepro.duplicate("not a dataframe")

def test_duplicate_invalid_parameters():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="keep"):
        prepro.duplicate(df, keep="invalid")
    with pytest.raises(ValueError, match="on_conflict"):
        prepro.duplicate(df, on_conflict="invalid")
    with pytest.raises(ValueError, match="flag_col"):
        prepro.duplicate(df, key="a", on_conflict="flag", flag_col=None)
    with pytest.raises(KeyError):
        prepro.duplicate(df, key="non_existent")

def test_duplicate_exact_full_row():
    # Remove exact full-row duplicates
    data = {
        "customer_id": ["C001", "C001", "C002"],
        "age": [25, 25, 30]
    }
    df = pd.DataFrame(data)
    
    # default keep="first"
    res_first = prepro.duplicate(df)
    assert len(res_first) == 2
    assert res_first.index.tolist() == [0, 2]
    
    # keep="last"
    res_last = prepro.duplicate(df, keep="last")
    assert len(res_last) == 2
    assert res_last.index.tolist() == [1, 2]
    
    # Original is not mutated
    assert len(df) == 3

def test_duplicate_exact_full_row_flag():
    data = {
        "customer_id": ["C001", "C001", "C002"],
        "age": [25, 25, 30]
    }
    df = pd.DataFrame(data)
    
    res = prepro.duplicate(df, flag_col="is_dup")
    assert len(res) == 3
    assert res["is_dup"].tolist() == [False, True, False]

def test_duplicate_key_level_drop():
    data = {
        "customer_id": ["C001", "C001", "C001", "C002"],
        "income": [45000, 45000, 50000, 60000]
    }
    df = pd.DataFrame(data)
    
    # keep="first"
    res_first = prepro.duplicate(df, key="customer_id", keep="first", on_conflict="drop")
    assert len(res_first) == 2
    assert res_first.loc[0, "income"] == 45000
    assert res_first["customer_id"].tolist() == ["C001", "C002"]

    # keep="last"
    res_last = prepro.duplicate(df, key="customer_id", keep="last", on_conflict="drop")
    assert len(res_last) == 2
    assert res_last.loc[2, "income"] == 50000
    assert res_last["customer_id"].tolist() == ["C001", "C002"]

def test_duplicate_key_level_flag():
    data = {
        "customer_id": ["C001", "C001", "C001", "C002"],
        "income": [45000, 45000, 50000, 60000]
    }
    df = pd.DataFrame(data)
    
    # Row 0 and Row 1 are exact duplicates. Row 2 is a near-duplicate of C001 (same key, different value).
    # Since C001 has conflicts, all C001 rows should be flagged True.
    # C002 has no conflict, so it should be flagged False.
    res = prepro.duplicate(df, key="customer_id", flag_col="is_near", on_conflict="flag")
    assert len(res) == 4
    assert res["is_near"].tolist() == [True, True, True, False]

def test_duplicate_key_level_flag_no_conflict():
    data = {
        "customer_id": ["C001", "C001", "C002"],
        "income": [45000, 45000, 60000]
    }
    df = pd.DataFrame(data)
    
    # C001 has duplicate rows but they are identical (no conflict on key-level values).
    # Therefore, no rows should be flagged as near-duplicate.
    res = prepro.duplicate(df, key="customer_id", flag_col="is_near", on_conflict="flag")
    assert len(res) == 3
    assert res["is_near"].tolist() == [False, False, False]

def test_duplicate_key_level_raise():
    # If there's a near-duplicate conflict, raise ValueError
    data_conflict = {
        "customer_id": ["C001", "C001", "C002"],
        "income": [45000, 50000, 60000]
    }
    df_conflict = pd.DataFrame(data_conflict)
    with pytest.raises(ValueError, match="Near-duplicates found"):
        prepro.duplicate(df_conflict, key="customer_id", on_conflict="raise")

    # If there are exact duplicates but no near-duplicates (no value mismatch), it should not raise
    data_no_conflict = {
        "customer_id": ["C001", "C001", "C002"],
        "income": [45000, 45000, 60000]
    }
    df_no_conflict = pd.DataFrame(data_no_conflict)
    res = prepro.duplicate(df_no_conflict, key="customer_id", on_conflict="raise")
    assert len(res) == 2
    assert res["customer_id"].tolist() == ["C001", "C002"]

def test_duplicate_composite_key():
    data = {
        "first_name": ["Alice", "Alice", "Alice", "Bob"],
        "dob": ["2000-01-01", "2000-01-01", "2000-01-01", "1990-05-05"],
        "income": [50000, 50000, 60000, 70000]
    }
    df = pd.DataFrame(data)
    
    # composite key has conflicts for Alice, 2000-01-01
    with pytest.raises(ValueError, match="Near-duplicates found"):
        prepro.duplicate(df, key=["first_name", "dob"], on_conflict="raise")
        
    res_drop = prepro.duplicate(df, key=["first_name", "dob"], keep="last", on_conflict="drop")
    assert len(res_drop) == 2
    assert res_drop.loc[2, "income"] == 60000
    assert res_drop["first_name"].tolist() == ["Alice", "Bob"]

def test_duplicate_keep_none_and_all():
    # 1. Exact full-row duplicates
    data = {
        "customer_id": ["C001", "C001", "C002"],
        "age": [25, 25, 30]
    }
    df = pd.DataFrame(data)
    
    # keep="none" -> drops both C001 rows
    res_none = prepro.duplicate(df, keep="none")
    assert len(res_none) == 1
    assert res_none["customer_id"].tolist() == ["C002"]
    
    # keep="all" -> retains everything
    res_all = prepro.duplicate(df, keep="all")
    assert len(res_all) == 3
    assert res_all["customer_id"].tolist() == ["C001", "C001", "C002"]
    
    # 2. Key level duplicates
    data_key = {
        "customer_id": ["C001", "C001", "C002"],
        "income": [45000, 50000, 60000]
    }
    df_key = pd.DataFrame(data_key)
    
    # keep="none" on key -> drops all copies of C001
    res_key_none = prepro.duplicate(df_key, key="customer_id", keep="none", on_conflict="drop")
    assert len(res_key_none) == 1
    assert res_key_none["customer_id"].tolist() == ["C002"]
    
    # keep="all" on key -> retains everything
    res_key_all = prepro.duplicate(df_key, key="customer_id", keep="all", on_conflict="drop")
    assert len(res_key_all) == 3
    assert res_key_all["customer_id"].tolist() == ["C001", "C001", "C002"]
