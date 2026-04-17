from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import logging

from src.config import OCSVM_KERNEL, OCSVM_NU, OCSVM_GAMMA, THRESHOLD_PERCENTILE

logger = logging.getLogger(__name__)

class OneClassSVMDetector:
    def __init__(self):
        self.pipeline = Pipeline([
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("ocsvm", OneClassSVM(kernel=OCSVM_KERNEL, nu=OCSVM_NU, gamma=OCSVM_GAMMA)),
        ])
        self.threshold: float | None = None

    def fit(self, features: np.ndarray) -> "OneClassSVMDetector":
        max_samples = min(5000, len(features))
        idx = np.random.RandomState(42).choice(len(features), max_samples, replace=False)
        subset = features[idx]

        logger.info(f"Training One-Class SVM | features={features.shape} | subset={subset.shape}")
        self.pipeline.fit(subset)
        scores = self.score(features)
        self.threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
        logger.info(f"OCSVM threshold: {self.threshold:.6f}")
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return -self.pipeline.decision_function(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Model not fitted")
        return (self.score(features) > self.threshold).astype(int)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path)
        logger.info(f"OneClassSVM saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "OneClassSVMDetector":
        data = joblib.load(path)
        detector = cls()
        detector.pipeline = data["pipeline"]
        detector.threshold = data["threshold"]
        return detector
