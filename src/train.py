from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import (
    SMAP_CHANNEL,
    WINDOW_SIZE,
    STEP_SIZE,
    ENSEMBLE_WEIGHTS,
    MODELS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.data_loader import load_entity, load_labels, sliding_windows, normalize
from src.features import extract_features
from src.models import IsolationForestDetector, OneClassSVMDetector, LSTMAutoencoderTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def evaluate_models(
    if_detector: IsolationForestDetector,
    ocsvm_detector: OneClassSVMDetector,
    lstm_trainer: LSTMAutoencoderTrainer,
    test_windows: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray | None,
) -> dict:

    if_scores_raw = if_detector.score(test_features)
    ocsvm_scores_raw = ocsvm_detector.score(test_features)
    lstm_scores_raw = lstm_trainer.score(test_windows)

    def minmax_norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-10)

    if_scores = minmax_norm(if_scores_raw)
    ocsvm_scores = minmax_norm(ocsvm_scores_raw)
    lstm_scores = minmax_norm(lstm_scores_raw)

    w_if, w_ocsvm, w_lstm = ENSEMBLE_WEIGHTS
    ensemble_scores = w_if * if_scores + w_ocsvm * ocsvm_scores + w_lstm * lstm_scores

    if_preds = if_detector.predict(test_features)
    ocsvm_preds = ocsvm_detector.predict(test_features)
    lstm_preds = lstm_trainer.predict(test_windows)

    votes = if_preds + ocsvm_preds + lstm_preds
    ensemble_preds = (votes >= 2).astype(int)

    results = {
        "if_anomaly_rate": float(if_preds.mean()),
        "ocsvm_anomaly_rate": float(ocsvm_preds.mean()),
        "lstm_anomaly_rate": float(lstm_preds.mean()),
        "ensemble_anomaly_rate": float(ensemble_preds.mean()),
    }

    if labels is not None:

        n_windows = len(test_windows)

        window_labels = np.array([
            int(labels[i * STEP_SIZE : i * STEP_SIZE + WINDOW_SIZE].any())
            for i in range(n_windows)
        ])

        for name, preds in [
            ("if", if_preds),
            ("ocsvm", ocsvm_preds),
            ("lstm", lstm_preds),
            ("ensemble", ensemble_preds),
        ]:
            results[f"{name}_precision"] = float(precision_score(window_labels, preds, zero_division=0))
            results[f"{name}_recall"] = float(recall_score(window_labels, preds, zero_division=0))
            results[f"{name}_f1"] = float(f1_score(window_labels, preds, zero_division=0))

        for name, scores in [
            ("if", if_scores),
            ("ocsvm", ocsvm_scores),
            ("lstm", lstm_scores),
            ("ensemble", ensemble_scores),
        ]:
            if window_labels.sum() > 0:
                results[f"{name}_roc_auc"] = float(roc_auc_score(window_labels, scores))

    return results

def train(entity: str = SMAP_CHANNEL) -> None:
    logger.info(f"Starting training pipeline for entity: {entity}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_data = load_entity(entity, split="train")
    test_data = load_entity(entity, split="test")
    labels = load_labels(entity)

    train_norm, test_norm, mean, std = normalize(train_data, test_data)
    logger.info(f"Data normalized | mean channels: {mean.shape}")

    train_windows = sliding_windows(train_norm, WINDOW_SIZE, STEP_SIZE)
    test_windows = sliding_windows(test_norm, WINDOW_SIZE, STEP_SIZE)
    logger.info(f"Windows: train={train_windows.shape}, test={test_windows.shape}")

    logger.info("Extracting statistical features...")
    t0 = time.time()
    train_features = extract_features(train_windows)
    test_features = extract_features(test_windows)
    logger.info(f"Feature extraction done in {time.time()-t0:.1f}s | shape={train_features.shape}")

    with mlflow.start_run(run_name=f"train_{entity}"):
        mlflow.log_params({
            "entity": entity,
            "window_size": WINDOW_SIZE,
            "step_size": STEP_SIZE,
            "train_windows": len(train_windows),
            "test_windows": len(test_windows),
            "n_features": train_features.shape[1],
            "ensemble_weights": str(ENSEMBLE_WEIGHTS),
        })

        logger.info("Training Isolation Forest...")
        t0 = time.time()
        if_detector = IsolationForestDetector()
        if_detector.fit(train_features)
        logger.info(f"IF trained in {time.time()-t0:.1f}s")

        if_path = MODELS_DIR / f"{entity}_isolation_forest.joblib"
        if_detector.save(if_path)
        mlflow.log_artifact(str(if_path))
        mlflow.log_param("if_n_estimators", 200)

        logger.info("Training One-Class SVM...")
        t0 = time.time()
        ocsvm_detector = OneClassSVMDetector()
        ocsvm_detector.fit(train_features)
        logger.info(f"OCSVM trained in {time.time()-t0:.1f}s")

        ocsvm_path = MODELS_DIR / f"{entity}_ocsvm.joblib"
        ocsvm_detector.save(ocsvm_path)
        mlflow.log_artifact(str(ocsvm_path))

        logger.info("Training LSTM Autoencoder (PyTorch)...")
        t0 = time.time()
        _, D = train_data.shape
        lstm_trainer = LSTMAutoencoderTrainer(input_dim=D, seq_len=WINDOW_SIZE)
        lstm_trainer.fit(train_windows)
        logger.info(f"LSTM trained in {time.time()-t0:.1f}s")

        lstm_path = MODELS_DIR / f"{entity}_lstm_autoencoder.pt"
        lstm_trainer.save(lstm_path)
        mlflow.log_artifact(str(lstm_path))

        for epoch, loss in enumerate(lstm_trainer.loss_history, 1):
            mlflow.log_metric("lstm_train_loss", loss, step=epoch)

        norm_path = MODELS_DIR / f"{entity}_norm_params.npz"
        np.savez(norm_path, mean=mean, std=std)
        mlflow.log_artifact(str(norm_path))

        logger.info("Evaluating on test data...")
        metrics = evaluate_models(
            if_detector, ocsvm_detector, lstm_trainer,
            test_windows, test_features, labels,
        )
        mlflow.log_metrics(metrics)

        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("=" * 60)

    logger.info("Training pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection models on NASA SMAP data")
    parser.add_argument("--entity", type=str, default=SMAP_CHANNEL, help="SMAP entity name (e.g. P-1, S-1)")
    args = parser.parse_args()
    train(args.entity)
