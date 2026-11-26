"""Summarise which clock CpGs are instrumentable, and how strongly.

Reads the GoDMC cis-mQTLs fetched by `fetch_mqtls.py` and asks, per clock:

  - what fraction of its CpGs have at least one genome-wide significant cis-mQTL
  - how many approximately independent instruments those CpGs have (GoDMC's clumped rows)
  - how strong the best instrument is, as an F statistic

F is (beta / se)^2 for a single SNP. The usual weak-instrument threshold is F below 10, but it
carries no information here and the output says so: instruments were pulled at p < 5e-8, which is
z of about 5.45, so every one of them already has F near 30 or above. The threshold is reported
only to make that redundancy visible rather than to let a reader mistake it for a filter that
did work.

The number that does carry information is the opposite one: the CpGs with no instrument at all.
A site that cannot be instrumented cannot be assessed by MR, in either direction. It is not
evidence of a non-causal site, it is absence of evidence, and any claim about how causally
enriched a clock is can only be made over the part of it that is assessable.

The comparison is the point. CausAge, DamAge and AdaptAge were selected by epigenome-wide MR,
which can only consider instrumentable CpGs. Horvath, Hannum and PhenoAge were fitted on
methylation alone with no genetic step.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import torch_stub  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from biolearn.model_gallery import ModelGallery  # noqa: E402

CLOCKS = ["Horvathv1", "Hannum", "PhenoAge", "YingCausAge", "YingDamAge", "YingAdaptAge"]
SELECTED_BY_MR = {"YingCausAge", "YingDamAge", "YingAdaptAge"}
CACHE = "data/godmc_cis_mqtls.json"


def f_stat(row):
    se = row.get("se")
    beta = row.get("beta_a1")
    if not se or beta is None:
        return np.nan
    return (float(beta) / float(se)) ** 2


def summarise(cpgs, mqtls):
    """Per-CpG instrument counts and best-instrument F."""
    rows = []
    for cpg in cpgs:
        hits = mqtls.get(cpg, [])
        clumped = [h for h in hits if h.get("clumped") == 1]
        fs = [f_stat(h) for h in clumped]
        fs = [f for f in fs if np.isfinite(f)]
        rows.append({"cpg": cpg, "any_mqtl": len(hits) > 0, "n_independent": len(clumped),
                     "best_f": max(fs) if fs else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/instrument_strength.json")
    args = ap.parse_args()

    if not os.path.exists(CACHE):
        sys.exit(f"missing {CACHE}: run scripts/fetch_mqtls.py first")
    mqtls = json.load(open(CACHE))
    gallery = ModelGallery()

    report, frames = {}, {}
    print(f"{'clock':14s} {'CpGs':>5s} {'instrumented':>14s} {'unassessable':>14s} "
          f"{'med instr':>9s} {'med best F':>10s}  selection")
    for name in CLOCKS:
        cpgs = sorted({c for c in gallery.get(name).coefficients.index if c != "intercept"})
        df = summarise(cpgs, mqtls)
        frames[name] = df
        strong = df.best_f >= 10
        entry = {
            "n_cpgs": int(len(df)),
            "with_mqtl": int(df.any_mqtl.sum()),
            "with_mqtl_pct": float(100 * df.any_mqtl.mean()),
            "strong_instrument": int(strong.sum()),
            "strong_instrument_pct": float(100 * strong.mean()),
            "median_independent": float(df.n_independent.median()),
            "median_best_f": float(np.nanmedian(df.best_f)),
            "selected_by_mr": name in SELECTED_BY_MR,
        }
        report[name] = entry
        entry["unassessable"] = int(len(df) - df.any_mqtl.sum())
        entry["unassessable_pct"] = float(100 * (1 - df.any_mqtl.mean()))
        print(f"{name:14s} {entry['n_cpgs']:5d} "
              f"{entry['with_mqtl']:5d} ({entry['with_mqtl_pct']:5.1f}%) "
              f"{entry['unassessable']:5d} ({entry['unassessable_pct']:5.1f}%) "
              f"{entry['median_independent']:9.1f} {entry['median_best_f']:10.1f}"
              f"  {'MR-selected' if entry['selected_by_mr'] else 'methylation only'}")

    mr = [n for n in CLOCKS if n in SELECTED_BY_MR]
    conv = [n for n in CLOCKS if n not in SELECTED_BY_MR]
    mr_pct = np.mean([report[n]["with_mqtl_pct"] for n in mr])
    conv_pct = np.mean([report[n]["with_mqtl_pct"] for n in conv])
    print(f"\nmean coverage: MR-selected {mr_pct:.1f}%  vs  methylation-only {conv_pct:.1f}%")
    print("\nThe 100% is not a finding: MR can only select instrumentable CpGs, so this is a check")
    print("that the pipeline does what it is meant to. The informative numbers are the background")
    print("rate for conventionally chosen sites, and the share of them MR cannot speak to at all.")
    report["_comparison"] = {"mr_selected_mean_pct": float(mr_pct),
                             "conventional_mean_pct": float(conv_pct)}

    os.makedirs("results", exist_ok=True)
    for name, df in frames.items():
        df.to_csv(f"results/instruments_{name}.csv", index=False)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
