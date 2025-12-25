import re

def clean_text(text: str) -> str:
    """
    Makale özetini temizler:
    - lowercase
    - noktalama işaretlerini kaldırır
    - fazla boşlukları temizler
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # lowercase
    text = text.lower()

    # noktalama temizleme
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # fazla boşluklar
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        raise ValueError("Text is empty after cleaning")

    return text
