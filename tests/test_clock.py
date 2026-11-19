"""Automated Unit and Integration Tests for the Epigenetic Aging Clock Pipeline."""

import os
import numpy as np

from src.data import MethylationCohortGenerator
from src.interpretability import annotate_cpg_weights, compute_pathway_enrichment
from src.metrics import compute_age_acceleration, evaluate_clock
from src.models import (
    DeepEpigeneticClock,
    ElasticNetClock,
    horvath_transform,
    inverse_horvath_transform,
)
from src.rejuvenation import evaluate_reprogramming_trajectory


def test_horvath_transform_inversion():
    """Verify mathematical exactness of Horvath forward and inverse transformations."""
    test_ages = np.array([0.2, 1.5, 12.0, 19.9, 20.0, 25.0, 45.0, 72.0, 95.0])
    transformed = horvath_transform(test_ages, adult_age=20.0)
    reconstructed = inverse_horvath_transform(transformed, adult_age=20.0)

    np.testing.assert_allclose(test_ages, reconstructed, rtol=1e-5, atol=1e-5)


def test_data_generator():
    """Verify synthetic multi-tissue cohort properties and biological ranges."""
    gen = MethylationCohortGenerator(seed=42)
    cohort = gen.generate_cohort(n_samples=50, n_background_cpgs=30)

    assert cohort.beta_values.shape == (50, 45)  # 15 canonical + 30 background
    assert np.all(cohort.beta_values >= 0.0)
    assert np.all(cohort.beta_values <= 1.0)
    assert len(cohort.chronological_age) == 50
    assert len(cohort.tissue) == 50
    assert not np.any(cohort.is_reprogrammed)


def test_elastic_net_convergence():
    """Test training and prediction of regularized ElasticNetClock."""
    gen = MethylationCohortGenerator(seed=42)
    cohort = gen.generate_cohort(n_samples=120, n_background_cpgs=40)

    clock = ElasticNetClock(random_state=42)
    clock.fit(cohort.beta_values, cohort.chronological_age, cpg_ids=cohort.cpg_ids)

    preds = clock.predict(cohort.beta_values)
    assert len(preds) == len(cohort.chronological_age)
    assert np.all(np.isfinite(preds))

    metrics = evaluate_clock(cohort.chronological_age, preds)
    assert metrics["mae_years"] < 8.0
    assert metrics["pearson_r"] > 0.85
    assert metrics["r2_score"] > 0.70

    selected_cpgs = clock.get_selected_cpgs()
    assert len(selected_cpgs) > 0
    # Canonical aging markers should dominate the top selected coefficients
    top_cpg_names = list(selected_cpgs.keys())[:10]
    assert any("cg16867657" in c or "cg24724428" in c for c in top_cpg_names)
    assert any("cg06639320" in c or "cg25547614" in c for c in top_cpg_names)


def test_deep_epigenetic_clock():
    """Test PyTorch DeepEpigeneticClock training and prediction."""
    gen = MethylationCohortGenerator(seed=42)
    cohort = gen.generate_cohort(n_samples=80, n_background_cpgs=20)

    deep_clock = DeepEpigeneticClock(epochs=20, hidden_dim=32, batch_size=32, seed=42)
    deep_clock.fit(cohort.beta_values, cohort.chronological_age, cpg_ids=cohort.cpg_ids)

    preds = deep_clock.predict(cohort.beta_values)
    assert preds.shape == cohort.chronological_age.shape
    assert np.all(np.isfinite(preds))

    importances = deep_clock.get_feature_importances()
    assert len(importances) == len(cohort.cpg_ids)
    assert abs(sum(importances.values()) - 1.0) < 1e-4


def test_age_acceleration_residuals():
    """Verify raw delta and regression-adjusted age acceleration."""
    y_true = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
    y_pred = np.array([22.0, 29.0, 44.0, 48.0, 65.0])

    raw_delta, adj_accel, summary = compute_age_acceleration(y_true, y_pred)
    assert len(raw_delta) == 5
    assert len(adj_accel) == 5
    # Adjusted residuals must sum to approximately 0 (mean = 0)
    assert abs(np.mean(adj_accel)) < 1e-4


def test_reprogramming_rejuvenation():
    """Verify Yamanaka factor cellular reprogramming resets biological clock."""
    gen = MethylationCohortGenerator(seed=42)
    train_cohort = gen.generate_cohort(n_samples=100, n_background_cpgs=20)

    clock = ElasticNetClock(random_state=42)
    clock.fit(train_cohort.beta_values, train_cohort.chronological_age, cpg_ids=train_cohort.cpg_ids)

    reprog_cohort = gen.generate_reprogramming_timecourse(
        donor_age=60.0, cpg_ids=train_cohort.cpg_ids, timepoints=[0, 3, 7, 21], replicates=3
    )

    summary_df = evaluate_reprogramming_trajectory(clock, reprog_cohort)
    assert len(summary_df) == 4

    # Day 0 should reflect ~60 years
    day0_age = summary_df.loc[summary_df["reprogramming_day"] == 0, "mean_bio_age"].values[0]
    assert day0_age > 50.0

    # Day 21 (iPSC) should be dramatically younger
    day21_age = summary_df.loc[summary_df["reprogramming_day"] == 21, "mean_bio_age"].values[0]
    assert day21_age < 15.0
    assert day21_age < day0_age - 35.0


if __name__ == "__main__":
    print("Running test_horvath_transform_inversion...")
    test_horvath_transform_inversion()
    print("Running test_data_generator...")
    test_data_generator()
    print("Running test_elastic_net_convergence...")
    test_elastic_net_convergence()
    print("Running test_deep_epigenetic_clock...")
    test_deep_epigenetic_clock()
    print("Running test_age_acceleration_residuals...")
    test_age_acceleration_residuals()
    print("Running test_reprogramming_rejuvenation...")
    test_reprogramming_rejuvenation()
    print("All tests passed successfully!")
