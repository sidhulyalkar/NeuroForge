# tests/test_endpoint.py

import pytest
from fastapi.testclient import TestClient
from middleware.endpoint.app import app

client = TestClient(app)


def test_predict_default():
    resp = client.post("/predict", json={"mode": "EEG"})
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_predict_invalid_mode():
    resp = client.post("/predict", json={"mode": "INVALID"})
    # We might treat invalid mode as 422 or as default EEG
    assert resp.status_code in (200, 422)
