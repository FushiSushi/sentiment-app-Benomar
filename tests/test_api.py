
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_available" in response.json()


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"texte": "Livraison rapide, très satisfait du produit"},
    )
    assert response.status_code == 200
    assert response.json()["label"] in ["positive", "negative"]
    assert 0 <= response.json()["score"] <= 1


def test_predict_empty_text():
    response = client.post("/predict", json={"texte": ""})
    assert response.status_code == 422
