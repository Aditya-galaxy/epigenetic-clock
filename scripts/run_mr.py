"""Two-sample Mendelian randomisation: methylation at each clock CpG on parental lifespan.

Exposure: GoDMC cis-mQTLs, clumped so the instruments per CpG are approximately independent.
Outcome: Timmers et al. 2019 combined parental lifespan (GWAS Catalog GCST006698).

Harmonisation is the step that decides whether any of this means anything, and it gets the most
care here:

  - The two studies report effects on whichever allele they chose. Where the outcome's effect
    allele is the exposure's other allele, the outcome beta's sign is flipped. Where the alleles
    do not correspond at all, the SNP is dropped rather than guessed at.
  - Palindromic SNPs (A/T and C/G) cannot be aligned by allele letters, because the pair reads the
    same on either strand. They can sometimes be resolved by allele frequency, but that fails near
    a frequency of 0.5 where the two possibilities are indistinguishable. Here they are dropped
    outright. Dropping costs power; guessing silently flips the sign of a causal estimate.

Estimators: the Wald ratio where a CpG has one instrument, inverse-variance weighted where it has
more. IVW assumes every instrument is valid, which cannot be checked with two or three SNPs, so
these estimates are a screen and not a verdict.

Multiple testing is handled by Benjamini-Hochberg across all CpGs tested.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import torch_stub  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from biolearn.model_gallery import ModelGallery  # noqa: E402

CLOCKS = ["Horvathv1", "Hannum", "PhenoAge", "YingCausAge", "YingDamAge", "YingAdaptAge"]
SELECTED_BY_MR = {"YingCausAge", "YingDamAge", "YingAdaptAge"}
MQTLS = "data/godmc_cis_mqtls.json"
OUTCOME = "data/parental_lifespan_instruments.tsv"
PALINDROMIC = {frozenset("AT"), frozenset("CG")}


def harmonise(exposure, outcome_row):
    """Align one instrument. Returns (beta_out, se_out) or None if it cannot be aligned."""
    e1, e2 = str(exposure["a1"]).upper(), str(exposure["a2"]).upper()
    o1, o2 = str(outcome_row["ALLELE1"]).upper(), str(outcome_row["ALLELE0"]).upper()
    if frozenset({e1, e2}) in PALINDROMIC:
        return None                       # strand-ambiguous; see module docstring
    beta, se = float(outcome_row["BETA"]), float(outcome_row["SE"])
    if {e1, e2} != {o1, o2}:
        return None                       # alleles do not correspond at all
    if e1 == o1:
        return beta, se
    if e1 == o2:
        return -beta, se
    return None


def mr_for_cpg(hits, outcome):
    """Wald ratio for one instrument, IVW for several."""
    betas, ses = [], []
    for h in hits:
        if h.get("clumped") != 1 or not h.get("rsid"):
            continue
        row = outcome.get(h["rsid"])
        if row is None:
            continue
        aligned = harmonise(h, row)
        if aligned is None:
            continue
        bo, so = aligned
        bx, sx = float(h["beta_a1"]), float(h["se"])
        if bx == 0 or sx == 0 or so == 0:
            continue
        betas.append(bo / bx)
        ses.append(so / abs(bx))          # first-order; ignores uncertainty in bx
    if not betas:
        return None
    betas, ses = np.array(betas), np.array(ses)
    w = 1 / ses ** 2
    est = float((betas * w).sum() / w.sum())
    se = float(np.sqrt(1 / w.sum()))
    z = est / se if se > 0 else np.nan
    return {"n_instruments": int(len(betas)), "estimate": est, "se": se,
            "pval": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/mr_parental_lifespan.json")
    args = ap.parse_args()

    for path in (MQTLS, OUTCOME):
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
    mqtls = json.load(open(MQTLS))
    od = pd.read_csv(OUTCOME, sep="\t")
    outcome = {r["SNP"]: r for _, r in od.iterrows()}
    print(f"outcome rows: {len(outcome)}")

    rows = []
    for cpg, hits in mqtls.items():
        res = mr_for_cpg(hits, outcome)
        if res:
            rows.append({"cpg": cpg, **res})
    df = pd.DataFrame(rows)
    df["qval"] = benjamini_hochberg(df.pval.values)
    df = df.sort_values("pval")
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/mr_per_cpg.csv", index=False)
    print(f"CpGs with a usable causal estimate: {len(df)}")
    print(f"  q < 0.05: {int((df.qval < 0.05).sum())}")

    gallery = ModelGallery()
    by_cpg = df.set_index("cpg")
    report = {}
    print(f"\n{'clock':14s} {'CpGs':>5s} {'tested':>13s} {'q<0.05':>12s}  selection")
    for name in CLOCKS:
        cpgs = {c for c in gallery.get(name).coefficients.index if c != "intercept"}
        sub = by_cpg.reindex([c for c in cpgs if c in by_cpg.index])
        sig = int((sub.qval < 0.05).sum())
        report[name] = {"n_cpgs": len(cpgs), "tested": int(len(sub)),
                        "tested_pct": float(100 * len(sub) / len(cpgs)),
                        "significant": sig,
                        "significant_pct_of_tested": float(100 * sig / len(sub)) if len(sub) else np.nan,
                        "selected_by_mr": name in SELECTED_BY_MR}
        r = report[name]
        print(f"{name:14s} {r['n_cpgs']:5d} {r['tested']:5d} ({r['tested_pct']:5.1f}%) "
              f"{sig:4d} ({r['significant_pct_of_tested']:5.1f}%)"
              f"  {'MR-selected' if r['selected_by_mr'] else 'methylation only'}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {args.out} and results/mr_per_cpg.csv")


if __name__ == "__main__":
    main()
