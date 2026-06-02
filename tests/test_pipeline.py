import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest import load_data, validate_data
from preprocess import preprocess
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# --- Ingest Tests ---
def test_load_data():
    df = load_data()
    assert len(df) == 284807
    assert 'Class' in df.columns

def test_validate_data():
    df = load_data()
    assert validate_data(df) == True

def test_class_distribution():
    df = load_data()
    fraud_rate = df['Class'].mean()
    assert fraud_rate < 0.01  # less than 1% fraud

# --- API Tests ---
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_legit_transaction():
    features = [0.1] * 30
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_predict_invalid_features():
    response = client.post("/predict", json={"features": [0.1, 0.2]})
    assert response.status_code == 400

def test_fraud_probability_range():
    features = [0.1] * 30
    response = client.post("/predict", json={"features": features})
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0