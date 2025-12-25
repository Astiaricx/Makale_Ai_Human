import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from scipy.sparse import hstack

from DatasetTemizleme import clean_text
from style_features import extract_style_features
from paths import DATA_DIR, MODELS_DIR

# =========================
# 1. Veriyi Yükle
# =========================
df = pd.read_csv(DATA_DIR / "İslenmis" / "dataset_combined.csv")

# =========================
# 2. Temizleme
# =========================
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =========================
# 3. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 4. TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# 5. STYLE FEATURES
# =========================
X_train_style = np.vstack(X_train.apply(extract_style_features))
X_test_style = np.vstack(X_test.apply(extract_style_features))

# =========================
# 6. FEATURE MERGE
# =========================
X_train_final = hstack([X_train_tfidf, X_train_style])
X_test_final = hstack([X_test_tfidf, X_test_style])

# =========================
# 7. GRID SEARCH
# =========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "kernel": ["linear"]
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_final, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# =========================
# 8. TEST SONUÇLARI
# =========================
y_pred = grid.best_estimator_.predict(X_test_final)
print(classification_report(y_test, y_pred))

# =========================
# 9. KAYDET
# =========================
joblib.dump(
    grid.best_estimator_,
    MODELS_DIR / "model_svm_style_tuned.pkl"
)
joblib.dump(
    vectorizer,
    MODELS_DIR / "tfidf_style.pkl"
)
