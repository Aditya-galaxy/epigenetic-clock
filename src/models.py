"""The clock itself: an elastic net on methylation beta values.

Horvath's age transform (logarithmic below adulthood, linear above) is included because it is the
convention, but both cohorts used here are adults, so it is off by default and its effect is
reported rather than assumed.
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ADULT = 20.0


def to_horvath(age):
    age = np.asarray(age, float)
    return np.where(age <= ADULT, np.log(age + 1) - np.log(ADULT + 1), (age - ADULT) / (ADULT + 1))


def from_horvath(z):
    z = np.asarray(z, float)
    return np.where(z <= 0, np.exp(z + np.log(ADULT + 1)) - 1, z * (ADULT + 1) + ADULT)


def clock(seed=0, l1_ratio=0.5, n_alphas=40, cv=5):
    """Elastic net with the penalty chosen by inner cross-validation on the training data.

    scikit-learn renamed `n_alphas` to `alphas` (an int means "this many, chosen automatically"),
    so both spellings are handled rather than pinning one version."""
    try:
        search = ElasticNetCV(l1_ratio=l1_ratio, alphas=n_alphas, cv=cv, max_iter=5000,
                              random_state=seed, n_jobs=-1, selection="random")
    except TypeError:
        search = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv, max_iter=5000,
                              random_state=seed, n_jobs=-1, selection="random")
    return make_pipeline(StandardScaler(), search)


def n_selected(fitted):
    return int((fitted[-1].coef_ != 0).sum())


def selected_probes(fitted, probe_names):
    coef = fitted[-1].coef_
    idx = np.flatnonzero(coef)
    order = idx[np.argsort(-np.abs(coef[idx]))]
    return [(probe_names[i], float(coef[i])) for i in order]
