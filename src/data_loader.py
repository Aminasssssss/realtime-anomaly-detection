from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from src.config import DATA_DIR, WINDOW_SIZE, STEP_SIZE

logger = logging.getLogger(__name__)

def load_entity(entity: str, split: str = "train") -> np.ndarray:
    path = DATA_DIR / split / f"{entity}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Entity file not found: {path}")
    data = np.load(path)
    logger.info(f"Loaded {split}/{entity}: shape={data.shape}")
    return data.astype(np.float32)

def load_labels(entity: str) -> Optional[np.ndarray]:
    csv_path = DATA_DIR / "labeled_anomalies.csv"
    if not csv_path.exists():
        logger.warning("labeled_anomalies.csv not found")
        return None

    df = pd.read_csv(csv_path)
    row = df[df["chan_id"] == entity]
    if row.empty:
        return None

    test_data = load_entity(entity, split="test")
    T = test_data.shape[0]
    labels = np.zeros(T, dtype=np.int8)

    sequences = eval(row["anomaly_sequences"].values[0])
    for start, end in sequences:
        labels[start : end + 1] = 1

    logger.info(f"Labels for {entity}: {labels.sum()} anomalous / {T} total")
    return labels

def sliding_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> np.ndarray:
    T, D = data.shape
    starts = range(0, T - window_size + 1, step_size)
    return np.stack([data[s : s + window_size] for s in starts], axis=0)

def normalize(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std if test_data is not None else None

    return train_norm, test_norm, mean.squeeze(), std.squeeze()

def get_available_entities() -> list[str]:
    train_dir = DATA_DIR / "train"
    if not train_dir.exists():
        return []
    return sorted([p.stem for p in train_dir.glob("*.npy")])
