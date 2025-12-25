import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from scipy.sparse import hstack

from DatasetTemizleme import clean_text
from style_features import extract_style_features

# =========================
# 1. Veriyi Yükle
# =========================
df = pd.read_csv("Veriler/İslenmis/dataset_combined.csv")

# =========================
# 2. Temizleme
# =========================
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =========================
# 3. Train / Test Split
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
# 4.5 STYLE FEATURES
# =========================
X_train_style = np.vstack(X_train.apply(extract_style_features))
X_test_style = np.vstack(X_test.apply(extract_style_features))

# =========================
# 4.6 FEATURE BİRLEŞTİRME
# =========================
X_train_final = hstack([X_train_tfidf, X_train_style])
X_test_final = hstack([X_test_tfidf, X_test_style])

# =========================
# 5. Logistic Regression
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)

# =========================
# 6. Değerlendirme
# =========================
y_pred = model.predict(X_test_final)
acc = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Accuracy (TF-IDF + Style): {acc:.4f}")

# =========================
# 7. Kaydet
# =========================
joblib.dump(model, "Modeller/model_logistic_style.pkl")
joblib.dump(vectorizer, "Modeller/tfidf.pkl")
