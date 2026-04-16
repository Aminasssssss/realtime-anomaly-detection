from __future__ import annotations

import numpy as np
from scipy import stats

def extract_features(windows: np.ndarray) -> np.ndarray:
    single = windows.ndim == 2
    if single:
        windows = windows[np.newaxis, ...]

    N, W, D = windows.shape
    feats = []

    feats.append(windows.mean(axis=1))
    feats.append(windows.std(axis=1))
    feats.append(windows.max(axis=1))
    feats.append(windows.min(axis=1))
    feats.append(windows.max(axis=1) - windows.min(axis=1))
    feats.append(stats.skew(windows, axis=1))
    feats.append(stats.kurtosis(windows, axis=1))

    diffs = np.diff(windows, axis=1)
    feats.append(diffs.mean(axis=1))
    feats.append(diffs.std(axis=1))
    feats.append(np.sqrt((windows ** 2).mean(axis=1)))

    signs = np.sign(windows)
    feats.append((np.diff(signs, axis=1) != 0).mean(axis=1))

    result = np.concatenate(feats, axis=1)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result[0] if single else result

def feature_names(D: int) -> list[str]:
    stats_names = ["mean", "std", "max", "min", "range", "skewness", "kurtosis", "diff_mean", "diff_std", "rms", "zcr"]
    return [f"ch{ch}_{stat}" for stat in stats_names for ch in range(D)]
