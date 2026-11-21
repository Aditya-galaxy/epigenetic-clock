"""Read Illumina methylation beta values and ages from GEO series matrix files.

A series matrix file is metadata lines beginning with `!`, then a probe-by-sample table between
`!series_matrix_table_begin` and `!series_matrix_table_end`. Ages live in the
`!Sample_characteristics_ch1` lines, whose wording differs between series, so the age field is
found by pattern rather than by position.

The 450K table is too large to hold in full, so `load_betas` streams it and keeps only the probes
asked for.
"""
import gzip
import re

import numpy as np
import pandas as pd

AGE_PATTERNS = (re.compile(r"\bage\s*\(?y?(?:ears?)?\)?\s*:\s*([0-9.]+)", re.I),
                re.compile(r"\bage\s*:\s*([0-9.]+)", re.I))


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def read_metadata(path):
    """Sample ids, ages, and every characteristics line, as a DataFrame indexed by sample id."""
    ids, char_lines = None, []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                ids = [f.strip('"') for f in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_lines.append([f.strip('"') for f in line.rstrip("\n").split("\t")[1:]])
    if ids is None:
        raise ValueError(f"no !Sample_geo_accession line in {path}")
    df = pd.DataFrame({f"char{i}": vals for i, vals in enumerate(char_lines)}, index=ids)
    df.index.name = "sample"
    df["age"] = [_find_age(row) for row in df.itertuples(index=False)]
    return df


MISSING = {"", "NA", "na", "NaN", "null", "NULL", "-"}


def _floats(rest):
    """One table row to float32. GEO files mark missing values as NA, null or an empty field."""
    parts = rest.rstrip("\n").split("\t")
    return np.array([np.nan if p in MISSING else p for p in parts], dtype=np.float32)


def _find_age(fields):
    for field in fields:
        if not isinstance(field, str):
            continue
        for pat in AGE_PATTERNS:
            m = pat.search(field)
            if m:
                return float(m.group(1))
    return np.nan


def probe_ids(path, limit=None):
    """Probe names in file order, read without loading the values."""
    out = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                next(fh)                                     # the sample-id header row
                break
        for line in fh:
            if line.startswith("!series_matrix_table_end"):
                break
            out.append(line.split("\t", 1)[0].strip('"'))
            if limit and len(out) >= limit:
                break
    return out


def scan_probe_stats(path, keep=None):
    """One streaming pass: per-probe mean, variance and missing count, without holding the table.

    Used to choose the most variable probes. Loading all 450,000 of them at once would need over a
    gigabyte; this keeps three numbers per probe instead."""
    keep = None if keep is None else set(keep)
    names, means, variances, missing = [], [], [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                next(fh)
                break
        for line in fh:
            if line.startswith("!series_matrix_table_end"):
                break
            name, _, rest = line.partition("\t")
            name = name.strip('"')
            if keep is not None and name not in keep:
                continue
            v = _floats(rest)
            names.append(name)
            means.append(np.nanmean(v))
            variances.append(np.nanvar(v))
            missing.append(int(np.isnan(v).sum()))
    return pd.DataFrame({"mean": means, "variance": variances, "missing": missing}, index=names)


def load_betas(path, keep=None, chunk=20000):
    """Beta values as a (samples x probes) DataFrame, keeping only probes in `keep`."""
    keep = None if keep is None else set(keep)
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
        header = next(fh).rstrip("\n").split("\t")
        samples = [f.strip('"') for f in header[1:]]
        frames, rows, names = [], [], []
        for line in fh:
            if line.startswith("!series_matrix_table_end"):
                break
            name, _, rest = line.partition("\t")
            name = name.strip('"')
            if keep is not None and name not in keep:
                continue
            rows.append(_floats(rest))
            names.append(name)
            if len(rows) >= chunk:
                frames.append(pd.DataFrame(np.vstack(rows), index=names, columns=samples))
                rows, names = [], []
        if rows:
            frames.append(pd.DataFrame(np.vstack(rows), index=names, columns=samples))
    if not frames:
        raise ValueError(f"no probes read from {path}")
    return pd.concat(frames).T


def load_cohort(path, keep=None):
    """Betas and ages for the samples that have both. Returns (X, age, metadata)."""
    meta = read_metadata(path)
    X = load_betas(path, keep=keep)
    meta = meta.reindex(X.index)
    ok = meta["age"].notna()
    return X.loc[ok], meta.loc[ok, "age"].astype(float), meta.loc[ok]
