"""
Iris dataset loader with Equal-Width discretization.

Equal-Width Binning
-------------------
Each continuous feature is divided into `n_bins` intervals of equal width:

    width = (max - min) / n_bins
    bin_i = floor((value - min) / width)  (clamped to [0, n_bins-1])

This converts each float into a discrete integer label (0, 1, 2, …) which
the ID3 engine treats as a categorical feature — producing a smaller, more
readable tree compared to using raw floats with binary threshold splits.

Usage
-----
    from src.ml.iris_loader import load_iris

    X_train, X_test, y_train, y_test, feature_names, label_names = load_iris()
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

# Path to the iris CSV relative to the repository root
_IRIS_PATH = Path(__file__).resolve().parents[2] / 'data' / 'iris.csv'

# Column names (after dropping the 'ID' column)
FEATURE_NAMES: list[str] = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
LABEL_NAME: str = 'class'
LABEL_NAMES: list[str] = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# ── Discretisation ────────────────────────────────────────────────────────────

def _compute_bins(
    values: list[float],
    n_bins: int,
) -> tuple[float, float, float]:
    """Return (min_val, max_val, width) for equal-width binning."""
    lo, hi = min(values), max(values)
    width = (hi - lo) / n_bins
    return lo, hi, width


def _discretise(value: float, lo: float, width: float, n_bins: int) -> int:
    """Map a continuous value to its bin index (0-indexed, clamped)."""
    if width == 0:
        return 0
    idx = int(math.floor((value - lo) / width))
    return max(0, min(n_bins - 1, idx))


def _bin_label(feat_name: str, bin_idx: int, lo: float, width: float) -> str:
    """Human-readable label for a bin, e.g. 'petallength_bin1 (2.97–4.93)'."""
    lo_edge = lo + bin_idx * width
    hi_edge = lo_edge + width
    return f'{feat_name}_bin{bin_idx} ({lo_edge:.2f}-{hi_edge:.2f})'


# ── Loading ───────────────────────────────────────────────────────────────────

def load_iris(
    path: str | Path | None = None,
    n_bins: int = 3,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[
    list[list[int]],   # X_train
    list[list[int]],   # X_test
    list[str],         # y_train
    list[str],         # y_test
    list[str],         # feature_names (discretised)
    list[str],         # label_names
]:
    """Load and discretise the Iris dataset, returning train/test splits.

    Parameters
    ----------
    path:
        Path to iris.csv. Defaults to ``data/iris.csv`` in the repo root.
    n_bins:
        Number of equal-width bins per continuous feature.
    test_ratio:
        Fraction of data to use for testing.
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, label_names
        - X_*       : lists of integer feature vectors (one int per feature)
        - y_*       : lists of class label strings
        - feature_names : one name per feature (e.g. 'petallength')
        - label_names   : sorted unique class names
    """
    path = Path(path) if path else _IRIS_PATH

    # ── Read CSV ──────────────────────────────────────────────────────────────
    raw_X: list[list[float]] = []
    raw_y: list[str] = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_X.append([
                float(row['sepallength']),
                float(row['sepalwidth']),
                float(row['petallength']),
                float(row['petalwidth']),
            ])
            raw_y.append(row['class'])

    n_samples  = len(raw_X)
    n_features = len(FEATURE_NAMES)

    # ── Compute per-feature bin parameters from the FULL dataset ─────────────
    bin_params: list[tuple[float, float, float]] = []  # (lo, hi, width) per feature
    for fi in range(n_features):
        col = [raw_X[i][fi] for i in range(n_samples)]
        lo, hi, width = _compute_bins(col, n_bins)
        bin_params.append((lo, hi, width))

    # ── Discretise ────────────────────────────────────────────────────────────
    disc_X: list[list[int]] = []
    for row in raw_X:
        disc_X.append([
            _discretise(row[fi], bin_params[fi][0], bin_params[fi][2], n_bins)
            for fi in range(n_features)
        ])

    # ── Shuffle & split ───────────────────────────────────────────────────────
    import random
    rng = random.Random(seed)
    indices = list(range(n_samples))
    rng.shuffle(indices)

    n_test  = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    X_train = [disc_X[i] for i in train_idx]
    y_train = [raw_y[i]  for i in train_idx]
    X_test  = [disc_X[i] for i in test_idx]
    y_test  = [raw_y[i]  for i in test_idx]

    label_names_out = sorted(set(raw_y))

    return X_train, X_test, y_train, y_test, FEATURE_NAMES, label_names_out


def load_iris_raw(
    path: str | Path | None = None,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Load the raw (continuous) Iris dataset without any discretisation.

    Returns
    -------
    (X, y, feature_names)
        X — list of 4-element float feature vectors
        y — list of class label strings
        feature_names — ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
    """
    path = Path(path) if path else _IRIS_PATH
    X: list[list[float]] = []
    y: list[str] = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([
                float(row['sepallength']),
                float(row['sepalwidth']),
                float(row['petallength']),
                float(row['petalwidth']),
            ])
            y.append(row['class'])

    return X, y, FEATURE_NAMES
