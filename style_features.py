import numpy as np
import re

def extract_style_features(text):

    if not isinstance(text, str):
        return np.zeros(6)

    if not text.strip():
        return np.zeros(6)

    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]

    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) == 0 or len(sentences) == 0:
        return np.zeros(6)

    sentence_lengths = [len(s.split()) for s in sentences]

    avg_sentence_length = np.mean(sentence_lengths)
    sentence_length_std = np.std(sentence_lengths)
    avg_word_length = np.mean([len(w) for w in words])

    punctuation_count = len(re.findall(r'[.,;:!?]', text))
    text_len = max(len(text), 1)
    punctuation_ratio = punctuation_count / text_len

    unique_word_ratio = len(set(words)) / len(words)
    word_repeat_ratio = 1 - unique_word_ratio

    return np.array([
        avg_sentence_length,
        sentence_length_std,
        avg_word_length,
        punctuation_ratio,
        unique_word_ratio,
        word_repeat_ratio
    ], dtype=float)
