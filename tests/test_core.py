import pytest
from prepro import clean_text

def test_clean_text_basic():
    assert clean_text("  Hello   World!  ") == "hello world!"

def test_clean_text_lowercase():
    assert clean_text("PrePro") == "prepro"

def test_clean_text_type_error():
    with pytest.raises(TypeError):
        clean_text(123)  # type: ignore
