import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from DatasetTemizleme import clean_text

BASE_DIR = Path(__file__).resolve().parents[1]


# =========================
# 1. Veri
# =========================

df = pd.read_csv(BASE_DIR / "Veriler" / "İslenmis" / "dataset_combined.csv")
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 2. TF-IDF
# =========================
vectorizer = joblib.load(BASE_DIR / "Modeller" / "tfidf.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# 3. Modeller
# =========================
models = {
    "Logistic Regression": BASE_DIR / "Modeller"/ "model_logistic_style_tuned.pkl",
    "SVM": BASE_DIR / "Modeller"/ "model_svm_style_tuned.pkl",
    "Random Forest": BASE_DIR / "Modeller"/ "model_rf_style_tuned.pkl"
}


# =========================
# 4. CONFUSION MATRIX
# =========================
for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])

    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"outputs/{name}_confusion_matrix.png")
    plt.close()

# =========================
# 5. ROC CURVE
# =========================
plt.figure()

for name, path in models.items():
    model = joblib.load(path)
    y_score = model.predict_proba(X_test_tfidf)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("outputs/roc_curve_comparison.png")
plt.close()
