"""Tests del análisis de significancia.

Sobre datos sintéticos pequeños, sin tocar los parquets reales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import precision_at_k
from src.evaluation.significance import (
    _descending_order,
    _ranking_hits,
    bootstrap_metric_distributions,
    confidence_intervals,
    overlaps,
    paired_wilcoxon,
)


def _synthetic(n: int = 300, n_pos: int = 80, seed: int = 0):
    """Devuelve (y_true, scores_perfecto, scores_aleatorio) indexados por cve_id."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[rng.choice(n, n_pos, replace=False)] = 1.0
    idx = pd.Index([f"CVE-{i}" for i in range(n)])
    y_true = pd.Series(y, index=idx)
    perfect = pd.Series(y, index=idx)  # el ranking ideal: positivos arriba
    noise = pd.Series(rng.random(n), index=idx)
    return y_true, perfect, noise


def test_ranking_hits_reproduce_precision():
    # Con conteos unitarios, la maquinaria del bootstrap == metrics.py.
    scores = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    order = _descending_order(scores)
    hits = _ranking_hits(y[order], np.ones(5), 2)
    assert hits / 2 == pytest.approx(precision_at_k(scores, y, 2))


def test_nan_va_al_fondo_en_el_orden():
    scores = np.array([np.nan, 5.0, 1.0])
    order = _descending_order(scores)
    assert order[-1] == 0  # el NaN queda el ultimo


def test_bootstrap_determinista_con_misma_semilla():
    y_true, perfect, noise = _synthetic()
    scores = {"FIS": perfect, "random": noise}
    d1 = bootstrap_metric_distributions(scores, y_true, ks=(50, 100), n_boot=500, seed=1)
    d2 = bootstrap_metric_distributions(scores, y_true, ks=(50, 100), n_boot=500, seed=1)
    for key in d1:
        assert np.array_equal(d1[key], d2[key])


def test_punto_dentro_del_ic():
    y_true, perfect, noise = _synthetic()
    scores = {"FIS": perfect, "random": noise}
    dists = bootstrap_metric_distributions(scores, y_true, ks=(50, 100), n_boot=800, seed=2)
    ci = confidence_intervals(dists, scores, y_true)
    for _, row in ci.iterrows():
        assert row["ci_lo"] <= row["point"] <= row["ci_hi"]


def test_metodos_identicos_no_significativos():
    y_true, perfect, _ = _synthetic()
    scores = {"FIS": perfect, "copia": perfect.copy()}
    dists = bootstrap_metric_distributions(scores, y_true, ks=(50, 100), n_boot=500, seed=3)
    wilc = paired_wilcoxon(dists, reference="FIS")
    copia = wilc[wilc["baseline"] == "copia"]
    assert (copia["diff_median"] == 0.0).all()
    assert (~copia["significant"]).all()
    assert (copia["wilcoxon_p"] == 1.0).all()


def test_metodo_dominante_tiene_ic_superior():
    y_true, perfect, noise = _synthetic()
    scores = {"perfect": perfect, "random": noise}
    dists = bootstrap_metric_distributions(scores, y_true, ks=(50,), n_boot=800, seed=4)
    ci = confidence_intervals(dists, scores, y_true)

    def bounds(method):
        r = ci[(ci["method"] == method) & (ci["metric"] == "P") & (ci["k"] == 50)]
        return float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0])

    perfect_lo, _ = bounds("perfect")
    _, random_hi = bounds("random")
    assert perfect_lo > random_hi
    # y por tanto sus IC no solapan
    assert not overlaps(ci, "perfect", "random", "P", 50)
