import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from scipy.sparse import hstack, csr_matrix

from paths import MODELS_DIR, DATA_DIR
from DatasetTemizleme import clean_text
from style_features import extract_style_features


# =========================
# 1. VERİYİ YÜKLE
# =========================
df = pd.read_csv(DATA_DIR / "İslenmis" / "dataset_combined.csv")
df["text"] = df["text"].astype(str)

# =========================
# 2. TEMİZLEME
# =========================
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =========================
# 3. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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
X_train_style = np.array([extract_style_features(t) for t in X_train])
X_test_style  = np.array([extract_style_features(t) for t in X_test])

X_train_style = csr_matrix(X_train_style)
X_test_style  = csr_matrix(X_test_style)

# =========================
# 6. FEATURE BİRLEŞTİRME
# =========================
X_train_final = hstack([X_train_tfidf, X_train_style])
X_test_final  = hstack([X_test_tfidf, X_test_style])

# =========================
# 7. GRID SEARCH
# =========================
param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [None, 30, 50],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train_final, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# =========================
# 8. TEST SET DEĞERLENDİRME
# =========================
best_rf = grid.best_estimator_

y_pred = best_rf.predict(X_test_final)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 9. KAYDET
# =========================
joblib.dump(best_rf, MODELS_DIR / "model_rf_style_tuned.pkl")
joblib.dump(vectorizer, MODELS_DIR / "tfidf.pkl")

print("\nTuned Random Forest modeli kaydedildi.")
