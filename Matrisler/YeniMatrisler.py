import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack

from DatasetTemizleme import clean_text
from style_features import extract_style_features
from paths import DATA_DIR, MODELS_DIR

# =========================
# OUTPUTS KLASÖRÜ
# =========================
OUTPUTS_DIR = Path("Outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# =========================
# 1. Veriyi Yükle
# =========================
df = pd.read_csv(DATA_DIR / "İslenmis" / "dataset_combined.csv")
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# =========================
# 2. Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. Modelleri Yükle
# =========================
vectorizer = joblib.load(MODELS_DIR / "tfidf_style.pkl")

models = {
    "logistic": joblib.load(MODELS_DIR / "model_logistic_style_tuned.pkl"),
    "svm": joblib.load(MODELS_DIR / "model_svm_style_tuned.pkl"),
    "random_forest": joblib.load(MODELS_DIR / "model_rf_style_tuned.pkl"),
}

# =========================
# 4. Feature Hazırla
# =========================
X_test_tfidf = vectorizer.transform(X_test)
X_test_style = np.vstack(X_test.apply(extract_style_features))
X_test_final = hstack([X_test_tfidf, X_test_style])

# =========================
# 5. CONFUSION MATRIX PNG
# =========================
for model_name, model in models.items():
    y_pred = model.predict(X_test_final)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Human", "AI"]
    )

    disp.plot()
    plt.title(f"{model_name.upper()} Confusion Matrix")

    output_file = OUTPUTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Oluşturuldu: {output_file}")
