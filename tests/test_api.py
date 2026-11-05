"""
tests/test_api.py
=================
FastAPI endpoint tests using pytest + httpx.

Run with:
    pytest tests/ -v

The tests use a lightweight mock artifact so they do NOT require a trained
model on disk — they are safe to run in CI before `python train.py`.
"""
import datetime as dt

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_mod


# ---------------------------------------------------------------------------
# Minimal mock artifact — mirrors the real artifact's structure
# ---------------------------------------------------------------------------

class _MockPipeline:
    classes_ = np.array(["High", "Low", "Moderate"])

    def __init__(self):
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        return np.full(len(X), "Low", dtype=object)

    def predict_proba(self, X):
        return np.tile([0.15, 0.70, 0.15], (len(X), 1))


class _MockRanker:
    """Ranks every observed value at 0.5 and keeps NaN, like QuantileTransformer."""

    def transform(self, df):
        values = df.to_numpy(dtype=float)
        return np.where(np.isnan(values), np.nan, 0.5)


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
    # Inject mock artifact fields directly; restored after the module's tests
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "_model",        _MOCK_ARTIFACT["model"], raising=False)
        mp.setattr(main_mod, "_ranker",       _MOCK_ARTIFACT["ranker"], raising=False)
        mp.setattr(main_mod, "_features",     _MOCK_ARTIFACT["features"], raising=False)
        mp.setattr(main_mod, "_classes",      _MOCK_ARTIFACT["classes"], raising=False)
        mp.setattr(main_mod, "_low_th",       _MOCK_ARTIFACT["low_th"], raising=False)
        mp.setattr(main_mod, "_high_th",      _MOCK_ARTIFACT["high_th"], raising=False)
        mp.setattr(main_mod, "_test_metrics", _MOCK_ARTIFACT["test_metrics"], raising=False)
        mp.setattr(main_mod, "_model_loaded", True)
        yield TestClient(main_mod.app)


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


class TestValidation:
    _BASE = {"date": "2020-01-15", "city": "Delhi", "PM2.5": 98.4}

    def test_invalid_date_422(self, client):
        r = client.post("/predict", json={**self._BASE, "date": "not-a-date"})
        assert r.status_code == 422

    def test_negative_concentration_422(self, client):
        r = client.post("/predict", json={**self._BASE, "PM2.5": -5})
        assert r.status_code == 422

    def test_field_name_accepted_as_well_as_alias(self, client):
        r = client.post("/predict", json={"date": "2020-01-15", "city": "Delhi",
                                          "PM2_5": 98.4, "PM10": 150, "NO2": 40})
        assert r.status_code == 200
        assert r.json()["daily_hrs"] == pytest.approx(0.5)

    def test_low_coverage_has_no_daily_hrs(self, client):
        # PM2.5 alone is 35% of the composite weight, below the 60% gate
        body = client.post("/predict", json=self._BASE).json()
        assert body["daily_hrs"] is None

    def test_duplicate_city_date_in_batch_422(self, client):
        r = client.post("/predict/batch", json={"readings": [self._BASE, self._BASE]})
        assert r.status_code == 422


class TestHistoryFeatures:
    """Batch readings of one city feed the rolling features, causally."""

    @staticmethod
    def _series(n_days, city="Delhi", start=dt.date(2020, 1, 1)):
        return [
            {"date": str(start + dt.timedelta(days=i)), "city": city,
             "PM2.5": 90, "PM10": 150, "NO2": 40, "SO2": 10, "CO": 1.5, "O3": 30}
            for i in range(n_days)
        ]

    def test_full_history_fills_rolling_features(self, client):
        body = client.post("/predict/batch", json={"readings": self._series(31)}).json()
        assert [r["history_days"] for r in body["results"]][:3] == [0, 1, 2]
        assert body["results"][-1]["history_days"] == 30

        X = main_mod._model.last_X
        assert X.iloc[-1][["hrs_mean_30", "hrs_lag_7", "hrs_std_14"]].notna().all()
        assert X.iloc[0][["hrs_mean_3", "hrs_lag_1"]].isna().all()

    def test_results_keep_request_order(self, client):
        readings = self._series(5)[::-1] + self._series(2, city="Patna")
        body = client.post("/predict/batch", json={"readings": readings}).json()
        assert [(r["city"], r["date"]) for r in body["results"]] == [
            (r["city"], r["date"]) for r in readings
        ]

    def test_cities_do_not_share_history(self, client):
        readings = self._series(10) + self._series(1, city="Patna",
                                                   start=dt.date(2020, 1, 10))
        body = client.post("/predict/batch", json={"readings": readings}).json()
        assert body["results"][-1]["history_days"] == 0

    def test_supplied_lag_used_only_without_history(self, client):
        client.post("/predict", json={**self._series(1)[0], "HRS_lag1": 0.9})
        assert main_mod._model.last_X.iloc[0]["hrs_lag_1"] == pytest.approx(0.9)

        readings = self._series(2)
        readings[1]["HRS_lag1"] = 0.9
        client.post("/predict/batch", json={"readings": readings})
        assert main_mod._model.last_X.iloc[1]["hrs_lag_1"] == pytest.approx(0.5)


def test_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(main_mod, "_model_loaded", False)
    monkeypatch.setattr(main_mod, "_load_error", "no artifact")
    c = TestClient(main_mod.app)
    assert c.get("/").status_code == 200
    assert c.get("/health").status_code == 503
    assert c.post("/predict", json={"date": "2020-01-15", "city": "Delhi"}).status_code == 503
