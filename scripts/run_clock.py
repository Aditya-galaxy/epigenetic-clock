"""Train the clock on one GEO cohort, test it on another, and write tables and figures.

    python scripts/run_clock.py [--train <series_matrix.txt.gz>] [--test ...] [--out results]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.pipeline import TEST, TRAIN, run


def plot_predictions(out_dir, report):
    within = pd.read_csv(f"{out_dir}/within_cohort_cv_predictions.csv")
    indep = pd.read_csv(f"{out_dir}/independent_cohort_predictions.csv")
    tr, te = report["cohorts"]["train"], report["cohorts"]["test"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    panels = [(axes[0], within, f"{tr['accession']}: cross-validated\nn={tr['n']}",
               report["within_cohort_cv"]),
              (axes[1], indep, f"{te['accession']}: independent cohort\nn={te['n']}",
               report["independent_cohort"])]
    lim = (min(within.age.min(), indep.age.min()) - 5, max(within.age.max(), indep.age.max()) + 5)
    for ax, df, title, sc in panels:
        ax.scatter(df.age, df.predicted, s=14, alpha=0.55, color="#2b5c8f", edgecolor="none")
        ax.plot(lim, lim, "k:", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Chronological age (years)")
        ax.set_title(f"{title}\nMAE {sc['mae']:.1f} y, r = {sc['pearson_r']:.2f}", fontsize=10)
    axes[0].set_ylabel("Predicted age (years)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/age_predictions.png", dpi=150)
    plt.close(fig)


def plot_bias(out_dir, report):
    """Error against age in both cohorts: the check that decides whether transfer or age is to blame."""
    within = pd.read_csv(f"{out_dir}/within_cohort_cv_predictions.csv")
    indep = pd.read_csv(f"{out_dir}/independent_cohort_predictions.csv")
    tr, te = report["cohorts"]["train"], report["cohorts"]["test"]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for df, label, colour, marker in ((within, f"{tr['accession']} (training, cross-validated)", "#999999", "o"),
                                      (indep, f"{te['accession']} (independent cohort)", "#c2543d", "^")):
        ax.scatter(df.age, df.predicted - df.age, s=16, alpha=0.6, color=colour,
                   marker=marker, edgecolor="none", label=label)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(40, ls=":", color="k", lw=0.9)
    bands = report["age_bands"]
    for label, x in (("under 40", 29), ("40 and over", 70)):
        parts = [f"{k.split()[0]} {v['bias']:+.1f}" for k, v in bands.get(label, {}).items()]
        ax.text(x, ax.get_ylim()[1] * 0.93, "\n".join(parts), fontsize=8, ha="center", va="top")
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("Predicted minus actual (years)")
    ax.set_title("The clock over-reads young people in both cohorts", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/bias_by_age.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default=TRAIN)
    ap.add_argument("--test", default=TEST)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    for p in (a.train, a.test):
        if not os.path.exists(p):
            sys.exit(f"missing {p}: see the README for the GEO download commands")
    report = run(out_dir=a.out, train_path=a.train, test_path=a.test)
    plot_predictions(a.out, report)
    plot_bias(a.out, report)
    print(f"\nwritten to {a.out}/", flush=True)


if __name__ == "__main__":
    main()
