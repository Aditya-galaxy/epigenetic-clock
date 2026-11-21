"""Scores, bootstrap intervals over samples, and the age-range confound check."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


def scores(age, pred):
    age, pred = np.asarray(age, float), np.asarray(pred, float)
    return {"n": int(len(age)), "mae": float(mean_absolute_error(age, pred)),
            "median_ae": float(np.median(np.abs(age - pred))),
            "pearson_r": float(pearsonr(age, pred).statistic),
            "bias": float(np.mean(pred - age))}


def bootstrap_mae(age, pred, n=5000, seed=0):
    age, pred = np.asarray(age, float), np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(age), (n, len(age)))
    maes = np.abs(age[idx] - pred[idx]).mean(1)
    return float(np.abs(age - pred).mean()), *np.percentile(maes, [2.5, 97.5]).tolist()


def baseline_mae(train_age, test_age):
    """Predicting the training mean for everyone: the number any clock must beat."""
    return float(np.mean(np.abs(np.asarray(test_age, float) - np.mean(train_age))))


def bootstrap_difference(a, b, n=5000, seed=0):
    """Mean(a) - mean(b) with a 95% interval, resampling each group independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = (a[rng.integers(0, len(a), (n, len(a)))].mean(1)
             - b[rng.integers(0, len(b), (n, len(b)))].mean(1))
    return float(a.mean() - b.mean()), *np.percentile(diffs, [2.5, 97.5]).tolist()


def age_acceleration(age, pred):
    """Residual from a least-squares fit of prediction on chronological age.

    The raw residual (pred - age) is correlated with age whenever the model regresses to the mean,
    so group comparisons use this instead."""
    age, pred = np.asarray(age, float), np.asarray(pred, float)
    slope, intercept = np.polyfit(age, pred, 1)
    return pred - (slope * age + intercept)
