from flask import Flask, render_template, request
import joblib
import numpy as np
from scipy.sparse import hstack

from DatasetTemizleme import clean_text
from style_features import extract_style_features
from paths import MODELS_DIR, TEMPLATES_DIR, STATIC_DIR

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)

# =========================
# MODELLERİ YÜKLE
# =========================
models = {
    "Logistic Regression": joblib.load(
        MODELS_DIR / "model_logistic_style_tuned.pkl"
    ),
    "SVM": joblib.load(
        MODELS_DIR / "model_svm_style_tuned.pkl"
    ),
    "Random Forest": joblib.load(
        MODELS_DIR / "model_rf_style_tuned.pkl"
    )
}

# Ortak TF-IDF
vectorizer = joblib.load(MODELS_DIR / "tfidf_style.pkl")

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    warning = None

    if request.method == "POST":
        text = request.form.get("text", "")

        if not text.strip():
            warning = "Lütfen bir metin giriniz."
        else:
            # 1️⃣ Temizleme
            cleaned = clean_text(text)

            # 2️⃣ TF-IDF
            X_tfidf = vectorizer.transform([cleaned])

            # 3️⃣ Style feature
            X_style = extract_style_features(cleaned).reshape(1, -1)

            # 4️⃣ Birleştir
            X_final = hstack([X_tfidf, X_style])

            # 5️⃣ 3 modelle tahmin
            results = {}

            for model_name, model in models.items():
                proba = model.predict_proba(X_final)[0]
                results[model_name] = {
                    "human": round(proba[0] * 100, 2),
                    "ai": round(proba[1] * 100, 2)
                }

    return render_template(
        "index.html",
        results=results,
        warning=warning
    )


if __name__ == "__main__":
    app.run(debug=True)
