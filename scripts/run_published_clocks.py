"""Run published clocks on both cohorts, and ask whether any of them escape the age artifact.

The earlier finding in this repository was that an elastic-net clock reads an independent cohort
about six years too old, and that this is not a cross-laboratory failure: the same bias appears
inside the training cohort's own cross-validation at the ages the test cohort is made of.

That raises a question this script answers. Is the bias a property of my clock, or of clocks that
regress on chronological age? Ying et al. (Nature Aging, 2024) built CausAge, DamAge and AdaptAge
from CpGs selected by epigenome-wide Mendelian randomisation, reporting that existing clocks are
not enriched for causal sites. If causal selection were the issue, CausAge should behave
differently. If the bias comes from the regression target and the training age distribution, it
should not.

Coefficients and transforms come from biolearn, so each clock is applied as its authors defined
it rather than as I guessed. Nothing here is fitted: these are published weights run on data they
did not see.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import torch_stub  # noqa: F401,E402  (must precede biolearn; see that module)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from biolearn.data_library import GeoData  # noqa: E402
from biolearn.model_gallery import ModelGallery  # noqa: E402

from src.data import load_cohort  # noqa: E402
from src.metrics import bootstrap_mae  # noqa: E402

TRAIN = "data/raw/GSE40279_series_matrix.txt.gz"
TEST = "data/raw/GSE41169_series_matrix.txt.gz"
CLOCKS = ["Horvathv1", "Hannum", "PhenoAge", "YingCausAge", "YingDamAge", "YingAdaptAge"]
CACHE = "data/published_clock_betas.npz"


def author_exclusions(meta):
    """Samples the depositors marked as not used in their own analysis."""
    flag = np.zeros(len(meta), dtype=bool)
    for col in meta.columns:
        if col == "age":
            continue
        vals = meta[col].astype(str).str.lower().str.replace(" ", "")
        flag |= vals.str.contains("used_in_analysis:no").values
    return pd.Series(flag, index=meta.index)


def needed_cpgs(gallery):
    cpgs = set()
    for name in CLOCKS:
        idx = gallery.get(name).coefficients.index
        cpgs |= {c for c in idx if c != "intercept"}
    return sorted(cpgs)


def load_both(cpgs):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"using cached betas from {CACHE}", flush=True)
        return (pd.DataFrame(z["Xtr"], index=z["tr_idx"], columns=z["cols"]),
                pd.Series(z["age_tr"], index=z["tr_idx"]),
                pd.DataFrame(z["Xte"], index=z["te_idx"], columns=z["cols"]),
                pd.Series(z["age_te"], index=z["te_idx"]))

    print(f"loading {len(cpgs)} CpGs from both cohorts (streams ~1.4 GB, takes minutes)", flush=True)
    Xtr, age_tr, _ = load_cohort(TRAIN, keep=cpgs)
    Xte, age_te, meta_te = load_cohort(TEST, keep=cpgs)

    excluded = author_exclusions(meta_te)
    if excluded.any():
        print(f"test: dropping {int(excluded.sum())} samples the depositors marked "
              f"'used_in_analysis: no'", flush=True)
        Xte, age_te = Xte.loc[~excluded], age_te.loc[~excluded]

    cols = [c for c in cpgs if c in Xtr.columns and c in Xte.columns]
    Xtr, Xte = Xtr.reindex(columns=cols), Xte.reindex(columns=cols)
    np.savez_compressed(CACHE, Xtr=Xtr.values, age_tr=age_tr.values,
                        tr_idx=Xtr.index.to_numpy(), Xte=Xte.values, age_te=age_te.values,
                        te_idx=Xte.index.to_numpy(), cols=np.array(cols))
    return Xtr, age_tr, Xte, age_te


def predict(model, X, fill_from):
    """Run one clock. Missing betas are filled with the training cohort's mean for that CpG.

    The clocks need every site they were built on, so a gap has to be filled with something. The
    training mean is the least informative choice available and is recorded in the report.
    """
    need = [c for c in model.coefficients.index if c != "intercept"]
    frame = X.reindex(columns=need)
    filled = int(frame.isna().sum().sum())
    frame = frame.fillna(fill_from.reindex(need))
    data = GeoData(metadata=pd.DataFrame(index=frame.index), dnam=frame.T)
    return model.predict(data)["Predicted"].reindex(frame.index).values, filled


def band_stats(age, pred, name):
    rows = []
    for label, sel in (("under 40", age < 40), ("40 and over", age >= 40)):
        if sel.sum() < 5:
            continue
        mae, lo, hi = bootstrap_mae(age[sel], pred[sel])
        rows.append({"clock": name, "band": label, "n": int(sel.sum()), "mae": mae,
                     "mae_ci": [lo, hi], "bias": float(np.mean(pred[sel] - age[sel]))})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/published_clocks.json")
    args = ap.parse_args()

    gallery = ModelGallery()
    cpgs = needed_cpgs(gallery)
    Xtr, age_tr, Xte, age_te = load_both(cpgs)
    means = Xtr.mean()
    print(f"train {Xtr.shape}, ages {age_tr.min():.0f}-{age_tr.max():.0f} (mean {age_tr.mean():.1f})")
    print(f"test  {Xte.shape}, ages {age_te.min():.0f}-{age_te.max():.0f} (mean {age_te.mean():.1f})")

    report = {"clocks": {}, "bands": [],
              "cohorts": {"train": {"accession": "GSE40279", "n": int(len(age_tr)),
                                    "age_mean": float(age_tr.mean()),
                                    "under_40": int((age_tr.values < 40).sum())},
                          "test": {"accession": "GSE41169", "n": int(len(age_te)),
                                   "age_mean": float(age_te.mean()),
                                   "under_40": int((age_te.values < 40).sum())}}}

    print(f"\n{'clock':14s} {'cohort':9s} {'n':>4s} {'MAE':>6s} {'bias':>7s} "
          f"{'r':>6s} {'<40 bias':>9s} {'40+ bias':>9s}")
    for name in CLOCKS:
        model = gallery.get(name)
        entry = {}
        for cohort, X, age in (("train", Xtr, age_tr.values), ("test", Xte, age_te.values)):
            pred, filled = predict(model, X, means)
            mae, lo, hi = bootstrap_mae(age, pred)
            bands = band_stats(age, pred, name)
            for b in bands:
                b["cohort"] = cohort
            report["bands"].extend(bands)
            by = {b["band"]: b["bias"] for b in bands}
            entry[cohort] = {"n": int(len(age)), "mae": mae, "mae_ci": [lo, hi],
                             "bias": float(np.mean(pred - age)),
                             "pearson_r": float(np.corrcoef(age, pred)[0, 1]),
                             "betas_filled": filled}
            print(f"{name:14s} {cohort:9s} {len(age):4d} {mae:6.2f} "
                  f"{entry[cohort]['bias']:+7.2f} {entry[cohort]['pearson_r']:6.3f} "
                  f"{by.get('under 40', float('nan')):+9.2f} "
                  f"{by.get('40 and over', float('nan')):+9.2f}")
            pd.DataFrame({"age": age, "predicted": pred}).to_csv(
                f"results/{name}_{cohort}.csv", index=False)
        report["clocks"][name] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
