"""
tests/test_api.py
=================
FastAPI endpoint tests using pytest + httpx.

Run with:
    pytest tests/ -v

The tests use a lightweight mock artifact so they do NOT require a trained
model on disk — they are safe to run in CI before `python train.py`.
"""
import types
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Minimal mock artifact — mirrors the real artifact's structure
# ---------------------------------------------------------------------------

class _MockPipeline:
    classes_ = ["High", "Low", "Moderate"]

    def predict(self, X):
        return np.array(["Low"])

    def predict_proba(self, X):
        return np.array([[0.15, 0.70, 0.15]])


class _MockRanker:
    def transform(self, df):
        import pandas as pd
        return pd.DataFrame(
            np.full((len(df), len(df.columns)), 0.5),
            columns=df.columns,
        )


_MOCK_FEATURES = [
    "hrs_mean_3", "hrs_mean_7", "hrs_mean_14", "hrs_mean_30",
    "hrs_std_7", "hrs_std_14",
    "hrs_lag_1", "hrs_lag_7",
    "hrs_trend",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "is_winter",
    "voc_index", "voc_observed", "weight_coverage",
]

_MOCK_ARTIFACT = {
    "model":        _MockPipeline(),
    "ranker":       _MockRanker(),
    "features":     _MOCK_FEATURES,
    "classes":      ["High", "Low", "Moderate"],
    "low_th":       0.4032,
    "high_th":      0.5935,
    "test_metrics": {"accuracy": 0.751, "macro_f1": 0.756},
}


# ---------------------------------------------------------------------------
# Patch the module before importing the app
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    import app.main as main_mod

    # Inject mock artifact fields directly
    main_mod._model        = _MOCK_ARTIFACT["model"]
    main_mod._ranker       = _MOCK_ARTIFACT["ranker"]
    main_mod._features     = _MOCK_ARTIFACT["features"]
    main_mod._classes      = _MOCK_ARTIFACT["classes"]
    main_mod._low_th       = _MOCK_ARTIFACT["low_th"]
    main_mod._high_th      = _MOCK_ARTIFACT["high_th"]
    main_mod._test_metrics = _MOCK_ARTIFACT["test_metrics"]
    main_mod._model_loaded = True

    return TestClient(main_mod.app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoot:
    def test_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_has_docs_key(self, client):
        assert "docs" in client.get("/").json()

    def test_model_loaded_true(self, client):
        assert client.get("/").json()["model_loaded"] is True


class TestHealth:
    def test_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_has_features(self, client):
        body = client.get("/health").json()
        assert "features" in body
        assert len(body["features"]) == len(_MOCK_FEATURES)

    def test_has_thresholds(self, client):
        body = client.get("/health").json()
        assert "risk_thresholds" in body
        assert body["risk_thresholds"]["low"] == pytest.approx(0.4032)

    def test_has_test_metrics(self, client):
        body = client.get("/health").json()
        assert "test_metrics" in body
        assert "macro_f1" in body["test_metrics"]


class TestPredict:
    _PAYLOAD = {
        "date": "2020-01-15",
        "city": "Delhi",
        "PM2.5": 98.4,
        "PM10": 152.0,
        "NO2": 43.1,
        "SO2": 11.2,
        "CO": 1.8,
        "O3": 29.5,
        "Benzene": 2.1,
        "Toluene": 4.3,
        "Xylene": 1.0,
    }

    def test_200(self, client):
        r = client.post("/predict", json=self._PAYLOAD)
        assert r.status_code == 200

    def test_response_schema(self, client):
        body = client.post("/predict", json=self._PAYLOAD).json()
        assert body["city"] == "Delhi"
        assert body["date"] == "2020-01-15"
        assert body["risk_level"] in {"Low", "Moderate", "High"}
        assert isinstance(body["probabilities"], dict)
        assert set(body["probabilities"].keys()) == {"High", "Low", "Moderate"}

    def test_probabilities_sum_to_one(self, client):
        body = client.post("/predict", json=self._PAYLOAD).json()
        total = sum(body["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_missing_pollutants_still_200(self, client):
        # Only mandatory fields; all pollutants omitted
        r = client.post("/predict", json={"date": "2020-06-01", "city": "Chennai"})
        assert r.status_code == 200

    def test_winter_date(self, client):
        payload = {**self._PAYLOAD, "date": "2020-12-20"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200

    def test_missing_date_422(self, client):
        r = client.post("/predict", json={"city": "Delhi"})
        assert r.status_code == 422

    def test_missing_city_422(self, client):
        r = client.post("/predict", json={"date": "2020-01-15"})
        assert r.status_code == 422


class TestPredictBatch:
    _PAYLOAD = {
        "readings": [
            {
                "date": "2020-01-15",
                "city": "Delhi",
                "PM2.5": 98.4,
                "PM10": 152.0,
                "NO2": 43.1,
            },
            {
                "date": "2020-01-16",
                "city": "Mumbai",
                "SO2": 7.0,
                "CO": 1.1,
                "O3": 22.0,
            },
            {
                "date": "2019-05-10",
                "city": "Bengaluru",
            },
        ]
    }

    def test_200(self, client):
        r = client.post("/predict/batch", json=self._PAYLOAD)
        assert r.status_code == 200

    def test_result_count(self, client):
        body = client.post("/predict/batch", json=self._PAYLOAD).json()
        assert len(body["results"]) == 3

    def test_each_has_risk_level(self, client):
        body = client.post("/predict/batch", json=self._PAYLOAD).json()
        for result in body["results"]:
            assert result["risk_level"] in {"Low", "Moderate", "High"}

    def test_cities_preserved(self, client):
        body = client.post("/predict/batch", json=self._PAYLOAD).json()
        cities = [r["city"] for r in body["results"]]
        assert cities == ["Delhi", "Mumbai", "Bengaluru"]

    def test_empty_readings_422(self, client):
        r = client.post("/predict/batch", json={"readings": []})
        assert r.status_code == 422
