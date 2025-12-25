import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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
# 7. SVM MODEL
# =========================
model = SVC(
    kernel="linear",
    C=1,
    probability=True
)

model.fit(X_train_final, y_train)

# =========================
# 8. Değerlendirme
# =========================
y_pred = model.predict(X_test_final)
acc = accuracy_score(y_test, y_pred)

print(f"SVM Accuracy (TF-IDF + Style): {acc:.4f}")

# =========================
# 9. Kaydet
# =========================
joblib.dump(model, MODELS_DIR / "model_svm_style.pkl")
joblib.dump(vectorizer, MODELS_DIR / "tfidf_style.pkl")
