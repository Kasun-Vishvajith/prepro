import pytest
import pandas as pd
import numpy as np
import prepro

def test_cast_requires_dataframe():
    with pytest.raises(TypeError):
        prepro.cast("not a dataframe", dtypes={"col": "Int64"})

def test_cast_basic_and_nullable():
    # Simple real example mimicking user description
    data = {
        "customer_id": ["C001", "C002", "C003"],
        "age": ["25", "31", "28"],
        "income": ["45000", "?", "62000"],
        "employment_type": ["full-time", "part-time", "full-time"],
        "default": [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    # Cast
    casted_df = prepro.cast(
        df,
        dtypes={
            "age": "Int64",
            "income": "Float64",
            "employment_type": "category",
            "default": "boolean",
        }
    )
    
    # Assert dtypes
    assert casted_df["age"].dtype == "Int64"
    assert casted_df["income"].dtype == "Float64"
    assert casted_df["employment_type"].dtype == "category"
    assert casted_df["default"].dtype == "boolean"
    
    # Assert missing value mapped properly
    assert pd.isna(casted_df.loc[1, "income"])
    
    # Original must not be mutated
    assert df["income"].iloc[1] == "?"
    assert df["age"].dtype != "Int64"

def test_cast_banking_missing_values():
    data = {
        "customer_id": ["C001", "C002", "C003", "C004"],
        "income": ["45000", "999", "9999", "62000"],
        "loan_type": ["personal", "N/A", "mortgage", "-"]
    }
    df = pd.DataFrame(data)
    
    casted_df = prepro.cast(
        df,
        dtypes={
            "income": "Float64",
            "loan_type": "category"
        },
        na_values=["?", "N/A", "-", "999", "9999"]
    )
    
    # "999" and "9999" in income should be converted to NaN
    assert pd.isna(casted_df.loc[1, "income"])
    assert pd.isna(casted_df.loc[2, "income"])
    assert casted_df.loc[0, "income"] == 45000.0
    
    # "N/A" and "-" in loan_type should be converted to NaN
    assert pd.isna(casted_df.loc[1, "loan_type"])
    assert pd.isna(casted_df.loc[3, "loan_type"])

def test_cast_medical_missing_values():
    data = {
        "age": ["45", ".", "50"],
        "diagnosis": ["A", "B", "."]
    }
    df = pd.DataFrame(data)
    
    casted_df = prepro.cast(
        df,
        dtypes={"age": "Int64", "diagnosis": "category"},
        na_values=["."]
    )
    
    assert pd.isna(casted_df.loc[1, "age"])
    assert pd.isna(casted_df.loc[2, "diagnosis"])

def test_cast_boolean_variations():
    data = {
        "col1": [True, False, None],
        "col2": ["1", "0", "None"],
        "col3": ["true", "false", "none"],
        "col4": ["yes", "no", "?"],
        "col5": ["T", "F", "-"]
    }
    df = pd.DataFrame(data)
    
    dtypes = {c: "boolean" for c in df.columns}
    casted_df = prepro.cast(df, dtypes=dtypes)
    
    for c in df.columns:
        assert casted_df[c].dtype == "boolean"
        assert casted_df.loc[0, c] == True
        assert casted_df.loc[1, c] == False
        assert pd.isna(casted_df.loc[2, c])

def test_cast_downcast():
    # Downcasting integers
    # Min/max fits in Int8
    data_int8 = {"col": [1, 2, 100, None]}
    df_int8 = pd.DataFrame(data_int8)
    res_int8 = prepro.cast(df_int8, dtypes={"col": "Int64"}, downcast=True)
    assert res_int8["col"].dtype == "Int8"
    
    # Min/max fits in Int16 but not Int8
    data_int16 = {"col": [1000, 2, 20000, None]}
    df_int16 = pd.DataFrame(data_int16)
    res_int16 = prepro.cast(df_int16, dtypes={"col": "Int64"}, downcast=True)
    assert res_int16["col"].dtype == "Int16"
    
    # Min/max fits in Int32 but not Int16
    data_int32 = {"col": [1000000, 2, 2000000, None]}
    df_int32 = pd.DataFrame(data_int32)
    res_int32 = prepro.cast(df_int32, dtypes={"col": "Int64"}, downcast=True)
    assert res_int32["col"].dtype == "Int32"
    
    # Float64 downcasts to Float32
    data_float = {"col": [1.5, 2.7, 3.8, None]}
    df_float = pd.DataFrame(data_float)
    res_float = prepro.cast(df_float, dtypes={"col": "Float64"}, downcast=True)
    assert res_float["col"].dtype == "Float32"
