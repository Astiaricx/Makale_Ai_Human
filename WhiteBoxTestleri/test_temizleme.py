import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DatasetTemizleme import clean_text

def test_clean_text_basic():
    text = "THIS Paper!!!"
    result = clean_text(text)
    assert result == "this paper"

def test_clean_text_empty_string():
    with pytest.raises(ValueError):
        clean_text("!!!")

def test_clean_text_non_string():
    with pytest.raises(ValueError):
        clean_text(123)
