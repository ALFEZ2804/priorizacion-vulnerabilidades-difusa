"""Tests de las métricas de ranking."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_positivos_arriba():
    scores = [10, 9, 8, 1, 0]
    y = [1, 1, 0, 0, 0]
    assert precision_at_k(scores, y, 2) == 1.0


def test_precision_positivos_abajo():
    scores = [0, 1, 2, 3, 4]  # los positivos tienen los scores mas bajos
    y = [1, 1, 0, 0, 0]
    assert precision_at_k(scores, y, 2) == 0.0


def test_precision_denominador_es_k():
    scores = [5, 4, 3, 2, 1]
    y = [1, 0, 0, 0, 0]
    assert precision_at_k(scores, y, 4) == pytest.approx(0.25)


def test_recall_completo_cuando_k_cubre_todo():
    scores = [5, 4, 3, 2, 1]
    y = [1, 0, 0, 0, 1]
    assert recall_at_k(scores, y, 5) == 1.0


def test_recall_parcial():
    scores = [5, 4, 3, 2, 1]
    y = [1, 0, 0, 0, 1]  # 2 positivos; el top-1 captura solo uno
    assert recall_at_k(scores, y, 1) == pytest.approx(0.5)


def test_recall_sin_positivos_es_cero():
    assert recall_at_k([3, 2, 1], [0, 0, 0], 2) == 0.0


def test_ndcg_orden_perfecto():
    scores = [3, 2, 1]
    y = [1, 1, 0]
    assert ndcg_at_k(scores, y, 3) == pytest.approx(1.0)


def test_ndcg_sin_positivos_es_cero():
    assert ndcg_at_k([1, 2, 3], [0, 0, 0], 3) == 0.0


def test_nan_scores_van_al_fondo():
    scores = [np.nan, 5.0, 1.0]
    y = [0, 1, 0]
    # el unico positivo (score 5) queda arriba; el NaN se hunde
    assert precision_at_k(scores, y, 1) == 1.0


def test_formas_distintas_lanzan_error():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1], 1)


def test_k_no_positivo_lanza_error():
    with pytest.raises(ValueError):
        recall_at_k([1, 2, 3], [1, 0, 0], 0)
