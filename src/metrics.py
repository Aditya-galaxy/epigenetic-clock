"""Evaluation Metrics and Age Acceleration Analysis.

Quantifies prediction accuracy, Pearson/Spearman correlation, and computes
raw and regression-adjusted Age Acceleration Residuals (AAR / IEAA).
"""

from typing import Dict, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


def evaluate_clock(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard geroscience accuracy and correlation metrics.

    Args:
        y_true: Chronological ages in years
        y_pred: Predicted biological epigenetic ages in years

    Returns:
        Dictionary of metrics
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)

    return {
        "mae_years": round(mae, 2),
        "medae_years": round(medae, 2),
        "rmse_years": round(rmse, 2),
        "r2_score": round(r2, 3),
        "pearson_r": round(float(pearson_r), 3),
        "pearson_p": float(pearson_p),
        "spearman_rho": round(float(spearman_rho), 3),
        "spearman_p": float(spearman_p),
    }


def compute_age_acceleration(
    chronological_age: np.ndarray, epigenetic_age: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Compute raw and regression-adjusted Age Acceleration Residuals.

    Raw ΔAge = Epigenetic Age - Chronological Age
    Regression-Adjusted (IEAA): Residuals from OLS regression of Epigenetic Age on Chronological Age.
    Removes edge regression-to-the-mean effects (Horvath 2013).

    Returns:
        raw_delta: (N,) raw delta age in years
        adjusted_acceleration: (N,) regression-adjusted acceleration residual
        summary_stats: dict of mean, std, accelerated count, decelerated count
    """
    y_true = np.asarray(chronological_age, dtype=float)
    y_pred = np.asarray(epigenetic_age, dtype=float)

    # 1. Raw Delta Age
    raw_delta = y_pred - y_true

    # 2. Regression-adjusted acceleration (IEAA)
    slope, intercept, r_val, p_val, std_err = stats.linregress(y_true, y_pred)
    expected_pred = intercept + slope * y_true
    adjusted_acceleration = y_pred - expected_pred

    n_samples = len(y_true)
    accelerated = int(np.sum(adjusted_acceleration > 3.0))
    decelerated = int(np.sum(adjusted_acceleration < -3.0))
    normal = n_samples - accelerated - decelerated

    summary = {
        "mean_raw_delta": round(float(np.mean(raw_delta)), 2),
        "std_raw_delta": round(float(np.std(raw_delta)), 2),
        "mean_adjusted_accel": round(float(np.mean(adjusted_acceleration)), 2),
        "std_adjusted_accel": round(float(np.std(adjusted_acceleration)), 2),
        "accelerated_samples": accelerated,
        "decelerated_samples": decelerated,
        "normal_samples": normal,
        "ols_slope": round(float(slope), 3),
        "ols_intercept": round(float(intercept), 2),
    }

    return raw_delta, adjusted_acceleration, summary
