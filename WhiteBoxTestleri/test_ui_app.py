import pytest

from app import app
from paths import MODELS_DIR
import joblib


@pytest.fixture
def client():
    """
    Flask test client.
    White-box: app iç yapısı bilinerek test ediliyor.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_ui_get_page_loads(client):
    """
    White-box:
    Ana route (/) çalışıyor mu?
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI / Human" in response.data


def test_ui_post_prediction_pipeline(client):
    """
    White-box:
    POST isteği -> clean_text -> tfidf -> model -> predict_proba zinciri
    """
    response = client.post(
        "/",
        data={
            "text": "This paper proposes a novel deep learning architecture.",
            "model": "logistic"
        }
    )

    assert response.status_code == 200

    html = response.data.decode("utf-8")
    assert "Human:" in html
    assert "AI:" in html


def test_ui_loaded_models_exist():
    """
    White-box:
    app.py içinde kullanılan model dosyaları gerçekten var mı?
    """
    assert (MODELS_DIR / "model_logistic_tuned.pkl").exists()
    assert (MODELS_DIR / "model_svm_tuned.pkl").exists()
    assert (MODELS_DIR / "model_rf_tuned.pkl").exists()
    assert (MODELS_DIR / "tfidf.pkl").exists()


def test_ui_model_predict_proba_shape():
    """
    White-box:
    Yüklenen model predict_proba çıktısı doğru mu?
    """
    model = joblib.load(MODELS_DIR / "model_logistic_tuned.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf.pkl")

    X = vectorizer.transform(["test text for probability"])
    proba = model.predict_proba(X)[0]

    assert len(proba) == 2
    assert round(sum(proba), 5) == 1.0

def test_warning_shown_when_text_is_empty(client):
    """
    White-box:
    Boş input durumunda uyarı mesajı gösteriliyor mu?
    """
    response = client.post("/", data={"text": ""})
    html = response.data.decode("utf-8")

    assert "Lütfen bir metin giriniz" in html
