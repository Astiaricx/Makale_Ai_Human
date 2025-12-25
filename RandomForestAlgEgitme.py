import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack, csr_matrix

from paths import MODELS_DIR, DATA_DIR
from DatasetTemizleme import clean_text
from style_features import extract_style_features


# =========================
# 1. VERİYİ YÜKLE
# =========================
df = pd.read_csv(DATA_DIR / "İslenmis" / "dataset_combined.csv")

# text kolonu %100 string olsun (ÇOK ÖNEMLİ)
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
# 5. STYLE FEATURES (KESİN DOĞRU HAL)
# =========================
X_train_style = np.array([extract_style_features(t) for t in X_train])
X_test_style  = np.array([extract_style_features(t) for t in X_test])

# sparse'a çevir
X_train_style = csr_matrix(X_train_style)
X_test_style  = csr_matrix(X_test_style)

# =========================
# 6. FEATURE BİRLEŞTİRME
# =========================
X_train_final = hstack([X_train_tfidf, X_train_style])
X_test_final  = hstack([X_test_tfidf, X_test_style])

# =========================
# 7. RANDOM FOREST MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_final, y_train)

# =========================
# 8. DEĞERLENDİRME
# =========================
y_pred = model.predict(X_test_final)
acc = accuracy_score(y_test, y_pred)

print(f"Random Forest (TF-IDF + Style) Accuracy: {acc:.4f}")

# =========================
# 9. KAYDET
# =========================
joblib.dump(model, MODELS_DIR / "model_rf_style.pkl")
joblib.dump(vectorizer, MODELS_DIR / "tfidf.pkl")

print("Model ve TF-IDF başarıyla kaydedildi.")
