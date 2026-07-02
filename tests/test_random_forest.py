"""Smoke tests del baseline Random Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.random_forest import FEATURES, rf_score, train_random_forest


def _synthetic_train(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """DataFrame sintético separable con la firma de `train.parquet`."""
    rng = np.random.default_rng(seed)
    # Mitad "peligrosos" (features altas, y_train=1), mitad benignos.
    pos = pd.DataFrame(
        {f: rng.uniform(0.6, 1.0, n // 2) for f in FEATURES}
    ).assign(y_train=1)
    neg = pd.DataFrame(
        {f: rng.uniform(0.0, 0.4, n // 2) for f in FEATURES}
    ).assign(y_train=0)
    df = pd.concat([pos, neg], ignore_index=True)
    df["target_year"] = rng.integers(2020, 2024, len(df))
    return df


def test_train_and_score_in_unit_interval():
    train = _synthetic_train()
    model = train_random_forest(train, random_state=0, n_iter=8)
    scores = rf_score(model, train)
    assert len(scores) == len(train)
    assert scores.between(0.0, 1.0).all()


def test_score_separates_classes():
    train = _synthetic_train()
    model = train_random_forest(train, random_state=0, n_iter=8)
    scores = rf_score(model, train)
    # Los positivos sintéticos deben puntuar, en media, por encima de los negativos.
    assert scores[train["y_train"] == 1].mean() > scores[train["y_train"] == 0].mean()


def test_handles_nan_without_error():
    train = _synthetic_train()
    model = train_random_forest(train, random_state=0, n_iter=8)
    df = train.head(5).copy()
    df.loc[df.index[0], "epss_percentile"] = np.nan
    scores = rf_score(model, df)  # el imputer debe absorber el NaN
    assert scores.notna().all()
    assert scores.between(0.0, 1.0).all()
