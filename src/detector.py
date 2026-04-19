from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import shap

from src.config import MODELS_DIR, ALERT_LOG_PATH, ENSEMBLE_WEIGHTS
from src.features import extract_features, feature_names
from src.models import IsolationForestDetector, OneClassSVMDetector, LSTMAutoencoderTrainer

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    timestamp: str
    is_anomaly: bool
    ensemble_score: float
    if_score: float
    ocsvm_score: float
    lstm_score: float
    if_pred: int
    ocsvm_pred: int
    lstm_pred: int
    votes: int
    top_features: list[dict]
    channel_errors: list[float]

class AnomalyDetector:
    def __init__(self, entity: str):
        self.entity = entity
        self._load_models()
        self._load_norm_params()
        self.feature_names_list: list[str] = []

    def _load_models(self):
        if_path = MODELS_DIR / f"{self.entity}_isolation_forest.joblib"
        ocsvm_path = MODELS_DIR / f"{self.entity}_ocsvm.joblib"
        lstm_path = MODELS_DIR / f"{self.entity}_lstm_autoencoder.pt"

        for p in [if_path, ocsvm_path, lstm_path]:
            if not p.exists():
                raise FileNotFoundError(f"Model not found: {p}\nRun: python -m src.train --entity {self.entity}")

        self.if_detector = IsolationForestDetector.load(if_path)
        self.ocsvm_detector = OneClassSVMDetector.load(ocsvm_path)
        self.lstm_trainer = LSTMAutoencoderTrainer.load(lstm_path)
        logger.info(f"All models loaded for entity '{self.entity}'")

    def _load_norm_params(self):
        norm_path = MODELS_DIR / f"{self.entity}_norm_params.npz"
        if norm_path.exists():
            data = np.load(norm_path)
            self.mean = data["mean"]
            self.std = data["std"]
        else:
            self.mean = None
            self.std = None

    def _normalize(self, window):
        if self.mean is not None:
            return (window - self.mean) / (self.std + 1e-10)
        return window

    def _shap_explanation(self, features):
        try:
            if not hasattr(self, "_shap_explainer"):
                background = shap.sample(features.reshape(1, -1), 100)
                self._shap_explainer = shap.KernelExplainer(
                    lambda x: -self.if_detector.model.score_samples(x),
                    background,
                )
            shap_values = np.array(self._shap_explainer.shap_values(features.reshape(1, -1), nsamples=50)).flatten()

            if not self.feature_names_list:
                D = features.shape[0] // 11
                self.feature_names_list = feature_names(max(D, 1))

            top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
            return [{
                "feature": self.feature_names_list[i] if i < len(self.feature_names_list) else f"feature_{i}",
                "shap_value": round(float(shap_values[i]), 4),
                "direction": "up" if shap_values[i] > 0 else "down",
            } for i in top_idx]
        except Exception as e:
            logger.debug(f"SHAP skipped: {e}")
            return []

    def _lstm_channel_errors(self, window):
        reconstruction = self.lstm_trainer.get_reconstruction(window)
        return [round(float(e), 6) for e in ((window - reconstruction) ** 2).mean(axis=0)]

    def predict(self, window_raw, run_shap=True):
        window_norm = self._normalize(window_raw)
        features = extract_features(window_norm)

        if_score = float(self.if_detector.score(features.reshape(1, -1))[0])
        ocsvm_score = float(self.ocsvm_detector.score(features.reshape(1, -1))[0])
        lstm_score = float(self.lstm_trainer.score(window_norm[np.newaxis, ...])[0])

        if_pred = int(if_score > self.if_detector.threshold)
        ocsvm_pred = int(ocsvm_score > self.ocsvm_detector.threshold)
        lstm_pred = int(lstm_score > self.lstm_trainer.threshold)
        votes = if_pred + ocsvm_pred + lstm_pred

        def soft(raw, thr):
            return float(raw / (thr + 1e-10))

        ensemble_score = (
            ENSEMBLE_WEIGHTS[0] * soft(if_score, self.if_detector.threshold)
            + ENSEMBLE_WEIGHTS[1] * soft(ocsvm_score, self.ocsvm_detector.threshold)
            + ENSEMBLE_WEIGHTS[2] * soft(lstm_score, self.lstm_trainer.threshold)
        )

        is_anomaly = votes >= 2

        result = AnomalyResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_anomaly=is_anomaly,
            ensemble_score=round(ensemble_score, 4),
            if_score=round(if_score, 6),
            ocsvm_score=round(ocsvm_score, 6),
            lstm_score=round(lstm_score, 6),
            if_pred=if_pred,
            ocsvm_pred=ocsvm_pred,
            lstm_pred=lstm_pred,
            votes=votes,
            top_features=self._shap_explanation(features) if run_shap and is_anomaly else [],
            channel_errors=self._lstm_channel_errors(window_norm),
        )

        if is_anomaly:
            self._log_alert(result)

        return result

    def _log_alert(self, result):
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        logger.warning(f"ANOMALY | score={result.ensemble_score:.3f} | votes={result.votes}/3")

    def health(self):
        return {
            "status": "healthy",
            "entity": self.entity,
            "models": ["isolation_forest", "one_class_svm", "lstm_autoencoder"],
            "if_threshold": round(self.if_detector.threshold, 6),
            "ocsvm_threshold": round(self.ocsvm_detector.threshold, 6),
            "lstm_threshold": round(self.lstm_trainer.threshold, 6),
        }

    def model_info(self):
        return {
            "entity": self.entity,
            "models": {
                "isolation_forest": {
                    "n_estimators": self.if_detector.model.n_estimators,
                    "contamination": self.if_detector.model.contamination,
                    "threshold": round(self.if_detector.threshold, 6),
                },
                "one_class_svm": {
                    "kernel": self.ocsvm_detector.pipeline["ocsvm"].kernel,
                    "nu": self.ocsvm_detector.pipeline["ocsvm"].nu,
                    "threshold": round(self.ocsvm_detector.threshold, 6),
                },
                "lstm_autoencoder": {
                    "input_dim": self.lstm_trainer.input_dim,
                    "seq_len": self.lstm_trainer.seq_len,
                    "threshold": round(self.lstm_trainer.threshold, 6),
                    "device": str(self.lstm_trainer.device),
                },
            },
            "ensemble": {
                "strategy": "weighted_soft_score + majority_voting",
                "weights": {"if": ENSEMBLE_WEIGHTS[0], "ocsvm": ENSEMBLE_WEIGHTS[1], "lstm": ENSEMBLE_WEIGHTS[2]},
                "voting_threshold": "2 out of 3",
            },
        }

