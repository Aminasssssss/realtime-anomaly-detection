

from __future__ import annotations

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.config import WINDOW_SIZE, SMAP_CHANNEL

@pytest.fixture
def mock_detector():
    
    detector = MagicMock()
    detector.health.return_value = {
        "status": "healthy",
        "entity": SMAP_CHANNEL,
        "models": ["isolation_forest", "one_class_svm", "lstm_autoencoder"],
        "if_threshold": 0.35,
        "ocsvm_threshold": 0.28,
        "lstm_threshold": 0.004,
    }
    detector.model_info.return_value = {
        "entity": SMAP_CHANNEL,
        "models": {
            "isolation_forest": {"n_estimators": 200, "contamination": 0.05, "threshold": 0.35},
            "one_class_svm": {"kernel": "rbf", "nu": 0.05, "threshold": 0.28},
            "lstm_autoencoder": {"input_dim": 25, "seq_len": 100, "threshold": 0.004, "device": "cpu"},
        },
        "ensemble": {
            "strategy": "weighted_soft_score + majority_voting",
            "weights": {"if": 0.25, "ocsvm": 0.25, "lstm": 0.50},
            "voting_threshold": "2 out of 3",
        },
    }
    return detector

@pytest.fixture
def mock_result_normal():
    from src.detector import AnomalyResult
    return AnomalyResult(
        timestamp="2026-04-20T10:00:00+00:00",
        is_anomaly=False,
        ensemble_score=0.42,
        if_score=0.21,
        ocsvm_score=0.19,
        lstm_score=0.002,
        if_pred=0,
        ocsvm_pred=0,
        lstm_pred=0,
        votes=0,
        top_features=[],
        channel_errors=[0.001, 0.002, 0.003],
    )

@pytest.fixture
def mock_result_anomaly():
    from src.detector import AnomalyResult
    return AnomalyResult(
        timestamp="2026-04-20T10:00:05+00:00",
        is_anomaly=True,
        ensemble_score=1.85,
        if_score=0.62,
        ocsvm_score=0.55,
        lstm_score=0.018,
        if_pred=1,
        ocsvm_pred=1,
        lstm_pred=1,
        votes=3,
        top_features=[
            {"feature": "ch0_std", "shap_value": 0.245, "direction": "up"},
            {"feature": "ch0_range", "shap_value": 0.189, "direction": "up"},
            {"feature": "ch3_mean", "shap_value": -0.134, "direction": "down"},
        ],
        channel_errors=[0.045, 0.012, 0.067, 0.003],
    )

@pytest.fixture
def client(mock_detector, mock_result_normal):
    
    import src.api as api_module

    api_module._detector = mock_detector
    api_module._simulator = MagicMock()
    mock_detector.predict.return_value = mock_result_normal

    with TestClient(api_module.app) as c:
        yield c

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "entity" in data
        assert "models" in data
        assert len(data["models"]) == 3

    def test_health_contains_thresholds(self, client):
        data = client.get("/health").json()
        assert "if_threshold" in data
        assert "ocsvm_threshold" in data
        assert "lstm_threshold" in data
        assert all(isinstance(data[k], float) for k in ["if_threshold", "ocsvm_threshold", "lstm_threshold"])

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_all_models(self, client):
        data = client.get("/model-info").json()
        assert "models" in data
        models = data["models"]
        assert "isolation_forest" in models
        assert "one_class_svm" in models
        assert "lstm_autoencoder" in models

    def test_model_info_ensemble_config(self, client):
        data = client.get("/model-info").json()
        ensemble = data["ensemble"]
        assert "weights" in ensemble
        assert sum(ensemble["weights"].values()) == pytest.approx(1.0, abs=0.01)

    def test_lstm_info_has_pytorch_device(self, client):
        data = client.get("/model-info").json()
        lstm = data["models"]["lstm_autoencoder"]
        assert "device" in lstm
        assert lstm["device"] in ["cpu", "cuda", "mps"]

class TestPredictEndpoint:
    def _make_window(self, D: int = 25, anomaly: bool = False) -> list[list[float]]:
        
        rng = np.random.default_rng(42 if not anomaly else 99)
        window = rng.normal(0, 1, (WINDOW_SIZE, D))
        if anomaly:

            window[50:60, 0] += 10.0
        return window.tolist()

    def test_predict_normal_window(self, client):
        window = self._make_window()
        response = client.post("/predict", json={"window": window, "run_shap": False})
        assert response.status_code == 200

    def test_predict_returns_correct_schema(self, client):
        window = self._make_window()
        data = client.post("/predict", json={"window": window, "run_shap": False}).json()

        required_fields = [
            "timestamp", "is_anomaly", "ensemble_score",
            "if_score", "ocsvm_score", "lstm_score",
            "if_pred", "ocsvm_pred", "lstm_pred", "votes",
            "top_features", "channel_errors",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_scores_are_floats(self, client):
        window = self._make_window()
        data = client.post("/predict", json={"window": window}).json()
        assert isinstance(data["ensemble_score"], float)
        assert isinstance(data["if_score"], float)
        assert isinstance(data["lstm_score"], float)

    def test_predict_votes_range(self, client):
        window = self._make_window()
        data = client.post("/predict", json={"window": window}).json()
        assert 0 <= data["votes"] <= 3

    def test_predict_anomaly_window_detected(self, client, mock_detector, mock_result_anomaly):
        mock_detector.predict.return_value = mock_result_anomaly
        window = self._make_window(anomaly=True)
        data = client.post("/predict", json={"window": window}).json()
        assert data["is_anomaly"] is True
        assert data["ensemble_score"] > 1.0

    def test_predict_anomaly_has_shap(self, client, mock_detector, mock_result_anomaly):
        mock_detector.predict.return_value = mock_result_anomaly
        window = self._make_window(anomaly=True)
        data = client.post("/predict", json={"window": window, "run_shap": True}).json()
        assert len(data["top_features"]) > 0
        first = data["top_features"][0]
        assert "feature" in first
        assert "shap_value" in first
        assert "direction" in first

    def test_predict_wrong_window_size(self, client):
        bad_window = [[0.0] * 25] * 50  # 50 instead of 100
        response = client.post("/predict", json={"window": bad_window})
        assert response.status_code == 422

    def test_predict_1d_window_rejected(self, client):
        bad_window = [0.0] * WINDOW_SIZE  # 1D instead of 2D
        response = client.post("/predict", json={"window": bad_window})
        assert response.status_code == 422

class TestAlertsEndpoint:
    def test_alerts_returns_200(self, client):
        response = client.get("/alerts")
        assert response.status_code == 200

    def test_alerts_empty_response(self, client, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "ALERT_LOG_PATH", tmp_path / "alerts.jsonl")
        import src.api as api_module
        response = client.get("/alerts")
        data = response.json()
        assert "alerts" in data
        assert "total" in data

    def test_alerts_last_n_param(self, client):
        response = client.get("/alerts?last_n=10")
        assert response.status_code == 200

class TestRootEndpoint:
    def test_root_returns_service_info(self, client):
        data = client.get("/").json()
        assert "service" in data
        assert "endpoints" in data
        assert "/predict" in data["endpoints"]
        assert "/stream" in data["endpoints"]

class TestFeatureExtraction:
    def test_extract_features_shape(self):
        from src.features import extract_features
        windows = np.random.randn(50, WINDOW_SIZE, 5)
        features = extract_features(windows)
        assert features.shape == (50, 5 * 11)  # 11 stats per channel

    def test_extract_features_single_window(self):
        from src.features import extract_features
        window = np.random.randn(WINDOW_SIZE, 5)
        features = extract_features(window)
        assert features.ndim == 1
        assert features.shape[0] == 5 * 11

    def test_features_no_nan(self):
        from src.features import extract_features
        windows = np.random.randn(20, WINDOW_SIZE, 3)
        features = extract_features(windows)
        assert not np.any(np.isnan(features))

class TestDataLoader:
    def test_sliding_windows_shape(self):
        from src.data_loader import sliding_windows
        data = np.random.randn(500, 5)
        windows = sliding_windows(data, window_size=100, step_size=10)
        assert windows.shape[1] == 100
        assert windows.shape[2] == 5

    def test_sliding_windows_count(self):
        from src.data_loader import sliding_windows
        data = np.random.randn(210, 3)
        windows = sliding_windows(data, window_size=100, step_size=10)
        expected = len(range(0, 210 - 100 + 1, 10))
        assert len(windows) == expected

    def test_normalize_zero_mean(self):
        from src.data_loader import normalize
        data = np.random.randn(100, 5) * 3 + 7
        norm, _, mean, std = normalize(data)
        assert np.allclose(norm.mean(axis=0), 0.0, atol=1e-5)

    def test_normalize_unit_std(self):
        from src.data_loader import normalize
        data = np.random.randn(100, 5) * 3 + 7
        norm, _, mean, std = normalize(data)
        assert np.allclose(norm.std(axis=0), 1.0, atol=1e-5)
