"""Evaluation metrics used by ReGeoDTA."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve


def get_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def get_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_squared_error(y_true, y_pred))


def get_pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def get_sd(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    if len(y_pred_array) < 2:
        return float("nan")

    regression = LinearRegression().fit(y_pred_array.reshape(-1, 1), y_true_array)
    fitted = regression.predict(y_pred_array.reshape(-1, 1))
    return float(np.sqrt(np.square(y_true_array - fitted).sum() / (len(y_pred_array) - 1)))


def get_aupr(y_true: Sequence[float], y_pred: Sequence[float], threshold: float = 7.0) -> float:
    y_true_binary = (np.asarray(y_true) >= threshold).astype(int)
    if np.unique(y_true_binary).size < 2:
        warnings.warn("AUPR is undefined because the labels contain only one class.", RuntimeWarning)
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred)
    return float(auc(recall, precision))


def r_squared_error(y_obs: Sequence[float], y_pred: Sequence[float]) -> float:
    y_obs_array = np.asarray(y_obs, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    y_obs_centered = y_obs_array - y_obs_array.mean()
    y_pred_centered = y_pred_array - y_pred_array.mean()
    denominator = np.square(y_obs_centered).sum() * np.square(y_pred_centered).sum()
    if denominator == 0:
        return float("nan")
    return float(np.square((y_obs_centered * y_pred_centered).sum()) / denominator)


def get_k(y_obs: Sequence[float], y_pred: Sequence[float]) -> float:
    y_obs_array = np.asarray(y_obs, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    denominator = np.square(y_pred_array).sum()
    return float((y_obs_array * y_pred_array).sum() / denominator) if denominator != 0 else float("nan")


def squared_error_zero(y_obs: Sequence[float], y_pred: Sequence[float]) -> float:
    y_obs_array = np.asarray(y_obs, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    k_value = get_k(y_obs_array, y_pred_array)
    denominator = np.square(y_obs_array - y_obs_array.mean()).sum()
    if denominator == 0 or np.isnan(k_value):
        return float("nan")
    return float(1 - np.square(y_obs_array - k_value * y_pred_array).sum() / denominator)


def get_rm2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    r2_value = r_squared_error(y_true, y_pred)
    r02_value = squared_error_zero(y_true, y_pred)
    if np.isnan(r2_value) or np.isnan(r02_value):
        return float("nan")
    return float(r2_value * (1 - np.sqrt(abs(r2_value**2 - r02_value**2))))


def get_cindex(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    concordant = 0.0
    comparable = 0
    for i in range(1, len(y_true)):
        for j in range(i):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            if y_true[i] > y_true[j]:
                concordant += float(y_pred[i] > y_pred[j]) + 0.5 * float(y_pred[i] == y_pred[j])
            else:
                concordant += float(y_pred[i] < y_pred[j]) + 0.5 * float(y_pred[i] == y_pred[j])
    return float(concordant / comparable) if comparable else 0.0


def compute_metrics(
    y_true: Sequence[float], y_pred: Sequence[float], aupr_threshold: float
) -> dict[str, float]:
    """Compute all metrics reported by the original training script."""
    return {
        "mse": get_mse(y_true, y_pred),
        "ci": get_cindex(y_true, y_pred),
        "mae": get_mae(y_true, y_pred),
        "sd": get_sd(y_true, y_pred),
        "pearson": get_pearson(y_true, y_pred),
        "rm2": get_rm2(y_true, y_pred),
        "aupr": get_aupr(y_true, y_pred, threshold=aupr_threshold),
    }


# Backward-compatible metric names.
get_MAE = get_mae
get_MSE = get_mse
get_CORR = get_pearson
get_SD = get_sd
