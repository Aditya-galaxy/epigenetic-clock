"""End-to-End Epigenetic Aging Clock Pipeline Orchestrator.

Trains regularized elastic-net and deep neural clocks, evaluates age acceleration,
performs genomic/pathway feature attribution, and benchmarks cellular reprogramming.
"""

import json
import os
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import MethylationCohortGenerator
from .interpretability import annotate_cpg_weights, compute_pathway_enrichment
from .metrics import compute_age_acceleration, evaluate_clock
from .models import DeepEpigeneticClock, ElasticNetClock
from .rejuvenation import evaluate_reprogramming_trajectory


class EpigeneticClockPipeline:
    """Full campaign orchestrator for epigenetic clock modeling and longevity benchmarks."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.generator = MethylationCohortGenerator(seed=seed)

    def run_campaign(
        self,
        n_samples: int = 500,
        n_background_cpgs: int = 250,
        test_size: float = 0.20,
        output_dir: str = "results",
    ) -> Dict:
        """Execute end-to-end training, cross-validation, and rejuvenation benchmarks."""
        os.makedirs(output_dir, exist_ok=True)
        print("=" * 70)
        print("  EPIGENETIC AGING CLOCK & CELLULAR REJUVENATION PIPELINE")
        print("=" * 70)

        # 1. Generate Multi-Tissue Cohort
        print(f"Generating synthetic multi-tissue cohort ({n_samples} donors, {n_background_cpgs + 15} CpGs)...")
        cohort = self.generator.generate_cohort(n_samples=n_samples, n_background_cpgs=n_background_cpgs)

        X = cohort.beta_values
        y = cohort.chronological_age
        cpg_ids = cohort.cpg_ids
        tissues = np.array(cohort.tissue)

        # 2. Train / Test Split
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, tissues, test_size=test_size, random_state=self.seed
        )
        print(f"Train set: {len(y_train)} donors | Held-out test set: {len(y_test)} donors")

        # 3. Train Canonical Elastic Net Clock
        print("\nTraining Canonical ElasticNet Epigenetic Clock (5-fold CV)...")
        enet_clock = ElasticNetClock(random_state=self.seed)
        enet_clock.fit(X_train, y_train, cpg_ids=cpg_ids)
        enet_preds = enet_clock.predict(X_test)
        enet_metrics = evaluate_clock(y_test, enet_preds)
        print(f"  ElasticNet Test MAE: {enet_metrics['mae_years']} yrs | Pearson r: {enet_metrics['pearson_r']} (R²: {enet_metrics['r2_score']})")

        # 4. Train Deep Epigenetic Clock (PyTorch)
        print("\nTraining DeepEpigeneticClock (Sparse Regularized MLP)...")
        deep_clock = DeepEpigeneticClock(epochs=150, seed=self.seed)
        deep_clock.fit(X_train, y_train, cpg_ids=cpg_ids)
        deep_preds = deep_clock.predict(X_test)
        deep_metrics = evaluate_clock(y_test, deep_preds)
        print(f"  Deep Clock Test MAE: {deep_metrics['mae_years']} yrs | Pearson r: {deep_metrics['pearson_r']} (R²: {deep_metrics['r2_score']})")

        # 5. Age Acceleration Analysis (on ElasticNet)
        raw_delta, adj_accel, accel_summary = compute_age_acceleration(y_test, enet_preds)
        print(f"\nAge Acceleration Residuals:")
        print(f"  Mean Raw ΔAge: {accel_summary['mean_raw_delta']} ± {accel_summary['std_raw_delta']} yrs")
        print(f"  Accelerated Donors (> +3 yrs): {accel_summary['accelerated_samples']} / {len(y_test)}")
        print(f"  Decelerated Donors (< -3 yrs): {accel_summary['decelerated_samples']} / {len(y_test)}")

        # 6. Genomic Interpretability & Pathway Enrichment
        selected_cpgs = enet_clock.get_selected_cpgs()
        print(f"\nFeature Selection: Clock isolated {len(selected_cpgs)} predictive CpGs out of {len(cpg_ids)} total.")
        annotated_df = annotate_cpg_weights(selected_cpgs, top_n=20)
        pathway_df = compute_pathway_enrichment(annotated_df)

        print("\nTop 5 Predictive CpG Biomarkers:")
        for idx, row in annotated_df.head(5).iterrows():
            print(f"  • {row['cpg_id']} ({row['gene']}): weight = {row['weight']:+.4f} | {row['pathway']}")

        # 7. Cellular Reprogramming Benchmark (OSKM Timecourse)
        print("\nBenchmarking Yamanaka Factor (OSKM) Cellular Rejuvenation Timecourse...")
        reprogramming_cohort = self.generator.generate_reprogramming_timecourse(
            donor_age=62.0, cpg_ids=cpg_ids, timepoints=[0, 3, 7, 12, 21], replicates=5
        )
        rejuvenation_summary = evaluate_reprogramming_trajectory(enet_clock, reprogramming_cohort)

        for _, row in rejuvenation_summary.iterrows():
            day = row["reprogramming_day"]
            bio_age = row["mean_bio_age"]
            rev = row["reversal_years"]
            pct = row["rejuvenation_pct"]
            stage = row["stage_label"]
            print(f"  Day {day:02d}: Bio Age = {bio_age:.1f} yrs (Reversed {rev:.1f} yrs / {pct:.1f}%) -> {stage}")

        # 8. Generate Visualizations
        self._plot_clock_accuracy(y_test, enet_preds, enet_metrics, t_test, os.path.join(output_dir, "clock_accuracy.png"))
        self._plot_interpretability(annotated_df, pathway_df, os.path.join(output_dir, "cpg_pathway_importance.png"))
        self._plot_reprogramming(rejuvenation_summary, os.path.join(output_dir, "rejuvenation_trajectory.png"))

        # 9. Export Summary JSON
        summary_payload = {
            "models": {
                "elastic_net": enet_metrics,
                "deep_clock": deep_metrics,
            },
            "age_acceleration": accel_summary,
            "cpg_count_total": len(cpg_ids),
            "cpg_count_selected": len(selected_cpgs),
            "top_cpg_biomarkers": annotated_df.head(10).to_dict(orient="records"),
            "pathway_enrichment": pathway_df.to_dict(orient="records"),
            "reprogramming_reversal": rejuvenation_summary.to_dict(orient="records"),
        }

        summary_path = os.path.join(output_dir, "clock_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)

        print("\n" + "=" * 70)
        print(f"Campaign Complete! Results and plots exported to '{output_dir}/'")
        print("=" * 70)
        return summary_payload

    def _plot_clock_accuracy(self, y_true, y_pred, metrics, tissues, output_path):
        """Scatter plot of chronological age vs. epigenetic predicted age."""
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        unique_tissues = np.unique(tissues)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tissues)))

        for tissue, color in zip(unique_tissues, colors):
            mask = tissues == tissue
            ax.scatter(y_true[mask], y_pred[mask], label=tissue.replace("_", " "), alpha=0.75, s=35, c=[color])

        # Identity line (y = x)
        min_val = 0.0
        max_val = max(np.max(y_true), np.max(y_pred)) + 5.0
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, alpha=0.7, label="Perfect Agreement (y = x)")

        # OLS regression fit line
        poly = np.polyfit(y_true, y_pred, 1)
        ax.plot([min_val, max_val], np.polyval(poly, [min_val, max_val]), "r-", lw=1.5, label=f"Fit (Slope: {poly[0]:.2f})")

        ax.set_title("Multi-Tissue Epigenetic Aging Clock Validation", fontsize=13, fontweight="bold")
        ax.set_xlabel("Chronological Age (Years)", fontsize=11)
        ax.set_ylabel("Predicted Epigenetic Age (Years)", fontsize=11)

        stats_text = (
            f"MAE: {metrics['mae_years']} yrs\n"
            f"Pearson r: {metrics['pearson_r']}\n"
            f"R² Score: {metrics['r2_score']}"
        )
        ax.text(
            0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.85)
        )

        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _plot_interpretability(self, annotated_df, pathway_df, output_path):
        """Top predictive CpG loci and functional pathway breakdown."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # Plot A: Top 10 CpGs by absolute weight
        top_cpgs = annotated_df.head(10).copy()
        colors = ["#d95f02" if w > 0 else "#7570b3" for w in top_cpgs["weight"]]
        labels = [f"{row['gene']} ({row['cpg_id'][:8]})" for _, row in top_cpgs.iterrows()]

        ax1.barh(labels[::-1], top_cpgs["weight"][::-1], color=colors[::-1])
        ax1.axvline(0, color="black", lw=0.8, linestyle="--")
        ax1.set_title("Top Epigenetic Clock CpG Biomarkers", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Elastic Net Coefficient (Weight)", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.5, axis="x")

        # Plot B: Pathway enrichment percentage
        top_pathways = pathway_df.head(6)
        ax2.barh(
            top_pathways["pathway"][::-1],
            top_pathways["relative_contribution_pct"][::-1],
            color="#1b9e77", alpha=0.85
        )
        ax2.set_title("Biological Pathway Enrichment (% Model Weight)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Relative Contribution (%)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="x")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _plot_reprogramming(self, rejuvenation_df, output_path):
        """Yamanaka factor (OSKM) partial reprogramming rejuvenation curve."""
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        days = rejuvenation_df["reprogramming_day"]
        bio_ages = rejuvenation_df["mean_bio_age"]
        stds = rejuvenation_df["std_bio_age"]

        ax.errorbar(days, bio_ages, yerr=stds, fmt="-o", color="#2b5c8f", lw=2.2, capsize=5, markersize=7, label="Epigenetic Age Trajectory")

        # Reference somatic baseline
        ax.axhline(62.0, color="#d95f02", linestyle="--", lw=1.2, label="Donor Baseline (Age 62)")
        # Embryonic iPSC line
        ax.axhline(0.0, color="#1b9e77", linestyle=":", lw=1.2, label="Embryonic State (Age 0)")

        # Shaded partial reprogramming safe zone (Days 3-7)
        ax.axvspan(3, 7, color="#e6ab02", alpha=0.18, label="Partial Reprogramming Window\n(Age Reversed, Identity Kept)")

        ax.set_title("Cellular Rejuvenation Kinetics via OSKM Induction", fontsize=13, fontweight="bold")
        ax.set_xlabel("Reprogramming Exposure (Days)", fontsize=11)
        ax.set_ylabel("Predicted Biological Age (Years)", fontsize=11)
        ax.set_xticks(days)
        ax.set_ylim(-5, 70)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
