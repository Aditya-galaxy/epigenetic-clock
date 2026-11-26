"""Are the MR-selected clocks' CpGs enriched for causal evidence on lifespan? And is it APOE?

Two questions, because the per-CpG results answer neither on their own.

First, individual CpGs. Six pass FDR across everything tested, and two of those are instrumented
from chr19 around 45 Mb, which is APOE: the strongest and most pleiotropic locus in human lifespan
genetics, acting through Alzheimer's disease and cardiovascular disease. A cis-mQTL there will
track lifespan whether or not methylation at that CpG does anything, which is horizontal
pleiotropy and not causal evidence. With the region excluded, nothing survives FDR at all.

Second, the aggregate. Individual significance is the wrong instrument for a question about sets,
so the rate of nominally significant results is compared against the 0.05 expected under the null.
That survives removing APOE.

The caveat that decides how to read the result is stated in the output rather than buried: Ying et
al. selected their CpGs by epigenome-wide MR against ageing-related outcomes. If lifespan or a
close correlate was among them, then finding lifespan enrichment among their CpGs re-finds their
selection criterion and is not independent evidence of anything.
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

MR_CLOCKS = ["YingCausAge", "YingDamAge", "YingAdaptAge"]
CONVENTIONAL = ["Horvathv1", "Hannum", "PhenoAge"]
APOE = (19, 44_400_000, 46_500_000)       # GRCh37, deliberately wide


def apoe_cpgs(mqtls, outcome):
    hit = set()
    for cpg, hits in mqtls.items():
        for h in hits:
            if h.get("clumped") != 1:
                continue
            row = outcome.get(h.get("rsid"))
            if row is None:
                continue
            if int(row["CHR"]) == APOE[0] and APOE[1] <= int(row["BP"]) <= APOE[2]:
                hit.add(cpg)
                break
    return hit


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/mr_enrichment.json")
    args = ap.parse_args()

    df = pd.read_csv("results/mr_per_cpg.csv").set_index("cpg")
    mqtls = json.load(open("data/godmc_cis_mqtls.json"))
    od = pd.read_csv("data/parental_lifespan_instruments.tsv", sep="\t").drop_duplicates("SNP")
    outcome = {r["SNP"]: r for _, r in od.iterrows()}

    # Restrict to CpGs that actually have an estimate; the mQTL cache covers more than that.
    apoe = apoe_cpgs(mqtls, outcome) & set(df.index)
    sig = set(df[df.qval < 0.05].index)
    clean = df.drop(index=sorted(apoe)).pval.dropna()
    q_clean = stats.false_discovery_control(clean.values)

    gallery = ModelGallery()

    def cpgs_of(names):
        out = set()
        for n in names:
            out |= {c for c in gallery.get(n).coefficients.index if c != "intercept"}
        return out & set(df.index)

    mr_set = cpgs_of(MR_CLOCKS)
    conv_set = cpgs_of(CONVENTIONAL) - mr_set   # the part not shared with an MR-selected clock

    report = {"per_cpg": {"tested": int(len(df)), "significant": len(sig),
                          "apoe_cpgs": len(apoe), "significant_in_apoe": len(apoe & sig),
                          "significant_without_apoe": int((q_clean < 0.05).sum()),
                          "tested_without_apoe": int(len(clean))},
              "enrichment": {}}

    print(f"tested CpGs: {len(df)}   instrumented from APOE: {len(apoe)}")
    print(f"pass q<0.05: {len(sig)}, of which {len(apoe & sig)} are APOE")
    print(f"pass q<0.05 with APOE removed: {int((q_clean < 0.05).sum())} of {len(clean)}")

    print("\nnominal p<0.05 rate (0.05 expected under the null):")
    for label, s in (("MR-selected", mr_set), ("conventional-only", conv_set)):
        for tag, keep in (("all", s), ("APOE excluded", s - apoe)):
            pv = df.loc[sorted(keep)].pval.dropna()
            k = int((pv < 0.05).sum())
            ci = stats.binomtest(k, len(pv), 0.05).proportion_ci(0.95)
            excl = ci.low > 0.05 or ci.high < 0.05
            print(f"  {label:18s} {tag:14s} n={len(pv):4d}  {k/len(pv):.3f} "
                  f"[{ci.low:.3f}, {ci.high:.3f}]  {'excludes 0.05' if excl else 'includes 0.05'}")
            report["enrichment"][f"{label}|{tag}"] = {
                "n": int(len(pv)), "rate": float(k / len(pv)),
                "ci": [float(ci.low), float(ci.high)], "excludes_null": bool(excl)}

    print("\nCAVEAT: Ying et al. selected these CpGs by epigenome-wide MR on ageing-related")
    print("outcomes. If lifespan or a correlate was among them, the enrichment above re-finds")
    print("their selection criterion rather than validating it independently.")

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
