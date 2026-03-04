from __future__ import annotations

import math
import numpy as np


def score_from_rmse(rmse: float | None) -> float | None:
    if rmse is None:
        return None
    if math.isnan(rmse):
        return None
    return float(1.0 / (1.0 + rmse))


def r2_score(y_true, y_pred) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return None
    return float(1.0 - ss_res / ss_tot)


def mape(y_true, y_pred, eps: float = 1e-6) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def regression_metrics(y_true, y_pred) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return None, None, None, None, None
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return None, None, None, None, None
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return mse, mae, rmse, r2, mape_val


def classification_metrics(y_true, y_pred, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {}
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {}
    # clip probabilities
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    # binary predictions
    y_hat = (y_pred >= threshold).astype(np.float64)
    # confusion
    tp = float(np.sum((y_hat == 1.0) & (y_true == 1.0)))
    tn = float(np.sum((y_hat == 0.0) & (y_true == 0.0)))
    fp = float(np.sum((y_hat == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((y_hat == 0.0) & (y_true == 1.0)))
    total = float(len(y_true))
    acc = (tp + tn) / total if total > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2.0 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) > 0 else None
    # losses
    bce = float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    brier = float(np.mean((y_pred - y_true) ** 2))
    brier_rmse = float(np.sqrt(brier))
    return {
        "bce": bce,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "brier": brier,
        "brier_rmse": brier_rmse,
    }
