"""Métricas de ranking para la evaluación.

Las tres métricas operan sobre un vector de `scores` (mayor = más prioritario) y
un vector binario `y_true` alineado por posición. Materializan la evaluación
retrospectiva del TFG: de las vulnerabilidades mejor puntuadas, cuántas entraron
realmente en KEV durante la ventana de test.

- `precision_at_k`: fracción de positivos entre los K mejor puntuados.
- `recall_at_k`: fracción de todos los positivos capturada en el top-K.
- `ndcg_at_k`: ganancia acumulada descontada normalizada, vía
  `sklearn.metrics.ndcg_score`.

Los NaN en `scores` se tratan como el peor valor posible (van al fondo del
ranking), de modo que las filas sin EPSS no contaminan el top-K de ningún método.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import ndcg_score


def _prepare(scores: ArrayLike, y_true: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Valida formas y envía los NaN de `scores` al fondo (valor finito mínimo)."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if scores.shape != y_true.shape:
        raise ValueError(
            f"scores y y_true deben tener la misma forma: {scores.shape} != {y_true.shape}"
        )
    if scores.ndim != 1:
        raise ValueError(f"scores debe ser 1-D, recibido ndim={scores.ndim}")
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        floor = np.nanmin(scores) - 1.0 if not nan_mask.all() else 0.0
        scores = np.where(nan_mask, floor, scores)
    return scores, y_true


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Índices de los `k` mayores scores (orden descendente estable)."""
    k = min(k, scores.shape[0])
    return np.argsort(-scores, kind="stable")[:k]


def precision_at_k(scores: ArrayLike, y_true: ArrayLike, k: int) -> float:
    """Precisión@K: positivos en el top-K dividido por K.

    Args:
        scores: Puntuaciones de cada elemento (mayor = más prioritario).
        y_true: Etiqueta binaria (1 = positivo) alineada con `scores`.
        k: Tamaño del corte superior.

    Returns:
        Precisión en [0, 1].
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    scores, y_true = _prepare(scores, y_true)
    idx = _top_k_indices(scores, k)
    return float(y_true[idx].sum() / k)


def recall_at_k(scores: ArrayLike, y_true: ArrayLike, k: int) -> float:
    """Recall@K: positivos en el top-K dividido por el total de positivos.

    Returns:
        Recall en [0, 1]; 0.0 si no hay positivos.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    scores, y_true = _prepare(scores, y_true)
    total = y_true.sum()
    if total == 0:
        return 0.0
    idx = _top_k_indices(scores, k)
    return float(y_true[idx].sum() / total)


def ndcg_at_k(scores: ArrayLike, y_true: ArrayLike, k: int) -> float:
    """NDCG@K con relevancias binarias.

    Returns:
        NDCG en [0, 1]; 0.0 si no hay positivos.
    """
    if k <= 0:
        raise ValueError("k debe ser positivo")
    scores, y_true = _prepare(scores, y_true)
    if y_true.sum() == 0:
        return 0.0
    return float(ndcg_score(y_true.reshape(1, -1), scores.reshape(1, -1), k=k))
