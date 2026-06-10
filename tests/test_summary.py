import pandas as pd
import pytest
from prepro.summary import summary

def test_summary_indexing():
    data = {
        "A": [1, 2, None, 4],
        "B": ["x", "?", "y", "z"],
    }
    df = pd.DataFrame(data)
    
    result = summary(df)
    
    # Check general summary via index 0
    assert result[0]["nrows"] == 4
    assert result[0]["ncolumns"] == 2
    assert result[0]["missing_rows"] == 1
    
    # Check detailed summary via index 1 (2D list)
    details = result[1]
    assert details[0] == ["Column", "Dtype", "Missing Count", "Missing Pct", "Unique Count", "Common Placeholders"]
    
    # Column A info
    assert details[1][0] == "A"
    assert details[1][2] == 1  # Missing count for A is 1
    
    # Column B info
    assert details[2][0] == "B"
    assert details[2][2] == 0  # Missing count for B is 0
    assert "'?': 1" in details[2][5]  # Placeholders

def test_summary_invalid_type():
    with pytest.raises(TypeError):
        summary([1, 2, 3])
