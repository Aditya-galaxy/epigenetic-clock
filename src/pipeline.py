"""Train a methylation clock on one cohort and test it on another.

The question is not whether a clock can fit its own cohort, which it always can, but whether it
still reads age correctly on blood samples processed by a different lab. The second cohort is also
younger and narrower in age, so the age-range confound is measured rather than ignored.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from .data import load_cohort, probe_ids, scan_probe_stats
from .metrics import age_acceleration, baseline_mae, bootstrap_difference, bootstrap_mae, scores
from .models import clock, n_selected, selected_probes

TRAIN = "data/raw/GSE40279_series_matrix.txt.gz"
TEST = "data/raw/GSE41169_series_matrix.txt.gz"
N_PROBES = 20000


def common_probes(train_path, test_path):
    a, b = probe_ids(train_path), set(probe_ids(test_path))
    return [p for p in a if p in b]


def pick_probes(stats, n=None):
    """Most variable complete probes in the training cohort.

    Selection uses no age labels, so it cannot leak the target into the cross-validation."""
    ok = stats[(stats["missing"] == 0) & stats["variance"].notna()]
    return list(ok["variance"].sort_values(ascending=False).head(n or N_PROBES).index)


def load_or_cache(train_path, test_path, cache="data/prepared.npz"):
    """Reading the gzipped tables takes minutes, so the prepared matrices are cached."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"using cached matrices from {cache}", flush=True)
        return (pd.DataFrame(z["Xtr"], index=z["tr_idx"], columns=z["probes"]),
                pd.Series(z["age_tr"], index=z["tr_idx"]),
                pd.DataFrame(z["Xte"], index=z["te_idx"], columns=z["probes"]),
                pd.Series(z["age_te"], index=z["te_idx"]),
                pd.DataFrame(z["meta_te"], index=z["te_idx"], columns=z["meta_cols"]),
                int(z["n_shared"]), int(z["n_excluded"]), int(z["n_absent"]))

    shared = common_probes(train_path, test_path)
    print(f"probes shared by both cohorts: {len(shared)}", flush=True)
    stats = scan_probe_stats(train_path, keep=shared)
    probes = pick_probes(stats)
    print(f"probes complete in training cohort: {int((stats['missing'] == 0).sum())}; "
          f"keeping the {len(probes)} most variable", flush=True)

    Xtr, age_tr, _ = load_cohort(train_path, keep=probes)
    Xtr = Xtr[probes]
    Xte, age_te, meta_te = load_cohort(test_path, keep=probes)
    excluded = _author_exclusions(meta_te)
    if excluded.any():
        print(f"test: dropping {int(excluded.sum())} samples the original authors marked "
              f"'used_in_analysis: no'", flush=True)
        Xte, age_te, meta_te = Xte.loc[~excluded], age_te.loc[~excluded], meta_te.loc[~excluded]
    Xte = Xte.reindex(columns=probes)
    absent = int(Xte.isna().all(axis=0).sum())
    Xte = Xte.fillna(Xtr.mean())

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, Xtr=Xtr.values, age_tr=age_tr.values, tr_idx=Xtr.index.to_numpy(),
                        Xte=Xte.values, age_te=age_te.values, te_idx=Xte.index.to_numpy(),
                        meta_te=meta_te.astype(str).values, meta_cols=meta_te.columns.to_numpy(),
                        probes=np.array(probes), n_shared=len(shared),
                        n_excluded=int(excluded.sum()), n_absent=absent)
    return Xtr, age_tr, Xte, age_te, meta_te, len(shared), int(excluded.sum()), absent


def run(out_dir="results", train_path=TRAIN, test_path=TEST, seed=0, folds=5):
    os.makedirs(out_dir, exist_ok=True)
    report = {}

    Xtr, age_tr, Xte, age_te, meta_te, n_shared, n_excluded, missing = load_or_cache(
        train_path, test_path)
    probes = list(Xtr.columns)
    report["author_exclusions_in_test"] = n_excluded
    report["probes_shared"] = n_shared
    print(f"train: {Xtr.shape[0]} samples, {Xtr.shape[1]} probes, "
          f"age {age_tr.min():.0f} to {age_tr.max():.0f}", flush=True)
    print(f"test:  {Xte.shape[0]} samples, age {age_te.min():.0f} to {age_te.max():.0f}, "
          f"{missing} probes absent and filled with the training mean", flush=True)

    report["cohorts"] = {
        "train": {"accession": os.path.basename(train_path).split("_")[0], "n": int(len(age_tr)),
                  "age_min": float(age_tr.min()), "age_max": float(age_tr.max()),
                  "age_mean": float(age_tr.mean())},
        "test": {"accession": os.path.basename(test_path).split("_")[0], "n": int(len(age_te)),
                 "age_min": float(age_te.min()), "age_max": float(age_te.max()),
                 "age_mean": float(age_te.mean())},
        "probes_used": int(Xtr.shape[1]), "probes_absent_in_test": missing}

    # 1. within-cohort, cross-validated
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred_cv = cross_val_predict(clock(seed=seed), Xtr.values, age_tr.values, cv=cv, n_jobs=1)
    m, lo, hi = bootstrap_mae(age_tr.values, pred_cv)
    report["within_cohort_cv"] = {**scores(age_tr.values, pred_cv), "mae_ci": [lo, hi],
                                  "baseline_mae_predict_mean": baseline_mae(age_tr.values, age_tr.values)}
    print(f"\nwithin-cohort {folds}-fold CV: MAE {m:.2f} [{lo:.2f}, {hi:.2f}] years, "
          f"r={report['within_cohort_cv']['pearson_r']:.3f}", flush=True)

    # 2. the clock itself, fitted on all of the training cohort
    model = clock(seed=seed).fit(Xtr.values, age_tr.values)
    report["model"] = {"probes_selected": n_selected(model), "alpha": float(model[-1].alpha_),
                       "l1_ratio": float(model[-1].l1_ratio)}
    pd.DataFrame(selected_probes(model, probes)[:25], columns=["probe", "coefficient"]).to_csv(
        f"{out_dir}/top_probes.csv", index=False)
    print(f"probes with non-zero weight: {report['model']['probes_selected']}", flush=True)

    # 3. independent cohort
    pred_te = model.predict(Xte.values)
    m2, lo2, hi2 = bootstrap_mae(age_te.values, pred_te)
    report["independent_cohort"] = {**scores(age_te.values, pred_te), "mae_ci": [lo2, hi2],
                                    "baseline_mae_predict_train_mean": baseline_mae(age_tr.values, age_te.values)}
    print(f"independent cohort:         MAE {m2:.2f} [{lo2:.2f}, {hi2:.2f}] years, "
          f"r={report['independent_cohort']['pearson_r']:.3f}, "
          f"bias {report['independent_cohort']['bias']:+.2f}", flush=True)
    print(f"  (predicting the training mean age for everyone would give MAE "
          f"{report['independent_cohort']['baseline_mae_predict_train_mean']:.2f})", flush=True)

    # 4. age-range confound: score the training cohort only where the two ranges overlap
    lo_a, hi_a = float(age_te.min()), float(age_te.max())
    inside = ((age_tr >= lo_a) & (age_tr <= hi_a)).values
    m3, lo3, hi3 = bootstrap_mae(age_tr.values[inside], pred_cv[inside])
    report["age_range_check"] = {"overlap_range": [lo_a, hi_a], "n_train_in_range": int(inside.sum()),
                                 "within_cohort_mae_in_range": m3, "mae_ci": [lo3, hi3]}
    print(f"\nage-range check: within-cohort CV restricted to {lo_a:.0f}-{hi_a:.0f} years "
          f"(n={int(inside.sum())}): MAE {m3:.2f} [{lo3:.2f}, {hi3:.2f}]", flush=True)

    # 4b. the same age bands in both cohorts. If a clock trained mostly on older people already
    # over-reads its own young samples, a young test cohort will look like a transfer failure.
    bands = {"under 40": lambda a: a < 40, "40 and over": lambda a: a >= 40}
    report["age_bands"] = {}
    print("\nage bands, same clock, both cohorts:", flush=True)
    print(f"  {'band':12s} {'cohort':14s} {'n':>4s} {'MAE':>6s} {'95% CI':>16s} {'bias':>7s}", flush=True)
    for label, fn in bands.items():
        row = {}
        for name, age_v, pred_v in (("training CV", age_tr.values, pred_cv),
                                    ("independent", age_te.values, pred_te)):
            sel = fn(age_v)
            if sel.sum() < 5:
                continue
            mm, ll, hh = bootstrap_mae(age_v[sel], pred_v[sel])
            bias = float(np.mean(pred_v[sel] - age_v[sel]))
            row[name] = {"n": int(sel.sum()), "mae": mm, "mae_ci": [ll, hh], "bias": bias}
            print(f"  {label:12s} {name:14s} {int(sel.sum()):4d} {mm:6.2f} "
                  f"  [{ll:5.2f}, {hh:5.2f}] {bias:+7.2f}", flush=True)
        report["age_bands"][label] = row

    # 5. group differences in the independent cohort, if its metadata names groups
    groups = _groups(meta_te)
    if groups is not None:
        accel = age_acceleration(age_te.values, pred_te)
        rows = [{"group": g, "n": int((groups == g).sum()),
                 "mean_age_acceleration_years": float(accel[(groups == g).values].mean())}
                for g in sorted(groups.dropna().unique())]
        report["groups_in_independent_cohort"] = rows
        print("\nage acceleration by group (independent cohort):", flush=True)
        for r in rows:
            print(f"  {r['group']:60s} n={r['n']:3d}  {r['mean_age_acceleration_years']:+.2f} y",
                  flush=True)
        if len(rows) == 2:
            a = accel[(groups == rows[0]["group"]).values]
            b = accel[(groups == rows[1]["group"]).values]
            d, lo_d, hi_d = bootstrap_difference(a, b)
            report["group_difference"] = {"groups": [rows[0]["group"], rows[1]["group"]],
                                          "difference_years": d, "ci": [lo_d, hi_d]}
            print(f"  difference: {d:+.2f} years [{lo_d:+.2f}, {hi_d:+.2f}]", flush=True)

    pd.DataFrame({"age": age_te.values, "predicted": pred_te}).to_csv(
        f"{out_dir}/independent_cohort_predictions.csv", index=False)
    pd.DataFrame({"age": age_tr.values, "predicted": pred_cv}).to_csv(
        f"{out_dir}/within_cohort_cv_predictions.csv", index=False)
    with open(f"{out_dir}/report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def _author_exclusions(meta):
    """Samples the depositors marked as not used in their own analysis."""
    flag = np.zeros(len(meta), dtype=bool)
    for col in meta.columns:
        if col == "age":
            continue
        vals = meta[col].astype(str).str.lower().str.replace(" ", "")
        flag |= vals.str.contains("used_in_analysis:no").values
    return pd.Series(flag, index=meta.index)


def _groups(meta):
    """A characteristics column that looks like a case/control label."""
    for col in meta.columns:
        if col == "age":
            continue
        vals = meta[col].astype(str)
        if vals.str.contains("diagnosis|disease|status|group", case=False).any() and 1 < vals.nunique() <= 6:
            return vals
    return None
