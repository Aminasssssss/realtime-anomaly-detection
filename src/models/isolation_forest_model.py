from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
import logging

from src.config import IF_N_ESTIMATORS, IF_CONTAMINATION, IF_MAX_SAMPLES, IF_RANDOM_STATE, THRESHOLD_PERCENTILE

logger = logging.getLogger(__name__)

class IsolationForestDetector:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            max_samples=IF_MAX_SAMPLES,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
        )
        self.threshold: float | None = None

    def fit(self, features: np.ndarray) -> "IsolationForestDetector":
        logger.info(f"Training Isolation Forest | features={features.shape}")
        self.model.fit(features)
        raw_scores = -self.model.score_samples(features)
        self.threshold = float(np.percentile(raw_scores, THRESHOLD_PERCENTILE))
        logger.info(f"IF threshold: {self.threshold:.6f}")
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Model not fitted")
        return (self.score(features) > self.threshold).astype(int)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)
        logger.info(f"IsolationForest saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "IsolationForestDetector":
        data = joblib.load(path)
        detector = cls()
        detector.model = data["model"]
        detector.threshold = data["threshold"]
        return detector
