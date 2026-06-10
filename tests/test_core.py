from prepro.core import clean_data


def test_clean_data():
    assert clean_data("test") == "test"
