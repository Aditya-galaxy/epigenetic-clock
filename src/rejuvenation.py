"""Cellular Rejuvenation and Reprogramming Benchmark.

Quantifies epigenetic age reversal in Yamanaka factor (OSKM) partial reprogramming
and targeted anti-aging interventions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .data import MethylationCohort


def evaluate_reprogramming_trajectory(model, cohort: MethylationCohort) -> pd.DataFrame:
    """Benchmark clock predictions across Yamanaka reprogramming timecourse (Days 0, 3, 7, 12, 21).

    Tracks the step-wise biological rejuvenation curve as somatic cells reprogram toward iPSCs.

    Args:
        model: Trained ElasticNetClock or DeepEpigeneticClock
        cohort: Reprogramming MethylationCohort instance

    Returns:
        DataFrame summarizing rejuvenation kinetics per timepoint
    """
    preds = model.predict(cohort.beta_values)
    df = cohort.to_dataframe()
    df["predicted_bio_age"] = preds

    # Extract day number from sample_ids (e.g. "OSKM_Day07_rep1" -> 7)
    days = []
    for sid in cohort.sample_ids:
        parts = sid.split("_")
        for p in parts:
            if p.startswith("Day"):
                days.append(int(p.replace("Day", "")))
                break
        else:
            days.append(0)

    df["reprogramming_day"] = days
    donor_baseline_age = float(np.mean(df[df["reprogramming_day"] == 0]["predicted_bio_age"]))

    # Summary by day
    summary_records = []
    for day, grp in df.groupby("reprogramming_day"):
        mean_pred = float(grp["predicted_bio_age"].mean())
        std_pred = float(grp["predicted_bio_age"].std())
        age_reversal_years = donor_baseline_age - mean_pred
        pct_rejuvenation = 100.0 * (age_reversal_years / (donor_baseline_age + 1e-6))

        summary_records.append({
            "reprogramming_day": int(day),
            "sample_count": len(grp),
            "mean_bio_age": round(mean_pred, 1),
            "std_bio_age": round(std_pred, 2),
            "reversal_years": round(max(0.0, age_reversal_years), 1),
            "rejuvenation_pct": round(max(0.0, pct_rejuvenation), 1),
            "stage_label": (
                "Somatic Baseline" if day == 0
                else ("Partial Reprogramming (Identity Preserved)" if day <= 7
                else "Intermediate Transition" if day < 21
                else "Fully Reprogrammed iPSC (Embryonic Reset)")
            ),
        })

    return pd.DataFrame(summary_records)
