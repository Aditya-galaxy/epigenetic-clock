"""Tests on a small synthetic series matrix (clearly fake) that check parsing and the metrics.

    python -m pytest tests -q
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_cohort, probe_ids, read_metadata  # noqa: E402
from src.metrics import age_acceleration, baseline_mae  # noqa: E402
from src.models import from_horvath, to_horvath  # noqa: E402

SAMPLES = ["GSM1", "GSM2", "GSM3"]
PROBES = {"cg001": [0.1, 0.5, 0.9], "cg002": [0.8, 0.4, 0.2], "cg003": ["NA", 0.3, 0.6]}


def _fake_series_matrix(path, ages=("age (y): 30", "age (y): 55", "age (y): 72")):
    with gzip.open(path, "wt") as fh:
        fh.write("!Series_title\t\"fake\"\n")
        fh.write("!Sample_geo_accession\t" + "\t".join(f'"{s}"' for s in SAMPLES) + "\n")
        fh.write("!Sample_characteristics_ch1\t" + "\t".join(f'"gender: F"' for _ in SAMPLES) + "\n")
        fh.write("!Sample_characteristics_ch1\t" + "\t".join(f'"{a}"' for a in ages) + "\n")
        fh.write("!series_matrix_table_begin\n")
        fh.write("\"ID_REF\"\t" + "\t".join(f'"{s}"' for s in SAMPLES) + "\n")
        for name, vals in PROBES.items():
            fh.write(f'"{name}"\t' + "\t".join(str(v) for v in vals) + "\n")
        fh.write("!series_matrix_table_end\n")
    return path


def test_metadata_finds_age_in_any_characteristics_line(tmp_path):
    p = _fake_series_matrix(str(tmp_path / "fake_series_matrix.txt.gz"))
    meta = read_metadata(p)
    assert list(meta.index) == SAMPLES
    assert meta["age"].tolist() == [30.0, 55.0, 72.0]


def test_probe_ids_and_beta_orientation(tmp_path):
    p = _fake_series_matrix(str(tmp_path / "fake_series_matrix.txt.gz"))
    assert probe_ids(p) == ["cg001", "cg002", "cg003"]
    X, age, _ = load_cohort(p, keep=["cg001", "cg002"])
    assert X.shape == (3, 2)                       # samples x probes
    assert list(X.index) == SAMPLES
    assert np.isclose(X.loc["GSM1", "cg001"], 0.1)
    assert age.tolist() == [30.0, 55.0, 72.0]


def test_missing_values_survive_as_nan(tmp_path):
    p = _fake_series_matrix(str(tmp_path / "fake_series_matrix.txt.gz"))
    X, _, _ = load_cohort(p, keep=["cg003"])
    assert np.isnan(X.loc["GSM1", "cg003"])
    assert X.dropna(axis=1).shape[1] == 0          # the probe is dropped, not silently filled


def test_samples_without_age_are_dropped(tmp_path):
    p = _fake_series_matrix(str(tmp_path / "f_series_matrix.txt.gz"),
                            ages=("age (y): 30", "tissue: blood", "age (y): 72"))
    X, age, _ = load_cohort(p, keep=["cg001"])
    assert list(X.index) == ["GSM1", "GSM3"] and age.tolist() == [30.0, 72.0]


def test_horvath_transform_round_trips():
    ages = np.array([1.0, 5.0, 20.0, 40.0, 90.0])
    assert np.allclose(from_horvath(to_horvath(ages)), ages)


def test_baseline_and_acceleration():
    train = np.array([40.0, 50.0, 60.0])
    assert np.isclose(baseline_mae(train, np.array([50.0])), 0.0)
    age = np.array([20.0, 40.0, 60.0, 80.0])
    assert np.allclose(age_acceleration(age, 0.5 * age + 10), 0, atol=1e-9)   # perfectly linear: no residual
