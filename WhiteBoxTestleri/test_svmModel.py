import sys
import os

import joblib
from DatasetTemizleme import clean_text
from paths import MODELS_DIR

def test_svm_prediction_output():
    model = joblib.load(MODELS_DIR/"model_svm.pkl")
    vectorizer = joblib.load(MODELS_DIR/"tfidf.pkl")

    text = "This paper presents a comprehensive analysis of quantum algorithms."
    cleaned = clean_text(text)

    X = vectorizer.transform([cleaned])
    prob = model.predict_proba(X)[0]

    assert len(prob) == 2
    assert round(sum(prob), 5) == 1.0
