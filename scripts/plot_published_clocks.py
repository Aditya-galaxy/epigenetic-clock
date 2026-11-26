"""Error against age on the independent cohort, for published clocks and for mine."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = [("Horvathv1", "Horvath (2013)", "#2b5c8f", "o"),
          ("Hannum", "Hannum (2013)", "#4e9a6a", "s"),
          ("YingCausAge", "CausAge (Ying 2024)", "#c2543d", "^")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

ax = axes[0]
mine = pd.read_csv("results/independent_cohort_predictions.csv")
ax.scatter(mine.age, mine.predicted - mine.age, s=20, alpha=0.65, color="#7a5195",
           edgecolor="none", label="elastic net trained on GSE40279")
ax.axhline(0, color="k", lw=1)
ax.axvline(40, ls=":", color="k", lw=0.9)
ax.set_title("A clock trained on a cohort with few young people\n"
             "reads this one about six years too old", fontsize=10)
ax.set_xlabel("Chronological age (years)")
ax.set_ylabel("Predicted minus actual (years)")
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
for name, label, colour, marker in SERIES:
    d = pd.read_csv(f"results/{name}_test.csv")
    bias = float(np.mean(d.predicted - d.age))
    ax.scatter(d.age, d.predicted - d.age, s=18, alpha=0.6, color=colour, marker=marker,
               edgecolor="none", label=f"{label}, bias {bias:+.1f} y")
ax.axhline(0, color="k", lw=1)
ax.axvline(40, ls=":", color="k", lw=0.9)
ax.set_title("Published clocks on the same samples:\n"
             "two are unbiased, the causality-enriched one is not", fontsize=10)
ax.set_xlabel("Chronological age (years)")
ax.legend(fontsize=8, loc="lower left")

for a in axes:
    a.set_ylim(-28, 28)
fig.suptitle("GSE41169, 89 blood samples, 75 of them under 40", fontsize=11, y=1.00)
fig.tight_layout()
fig.savefig("results/published_clocks_bias.png", dpi=150, bbox_inches="tight")
print("written results/published_clocks_bias.png")
