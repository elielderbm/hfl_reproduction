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
