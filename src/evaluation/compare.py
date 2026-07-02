"""Tabla comparativa de métodos de priorización.

Dado un conjunto de scores (uno por método) y el ground truth `y_true`, calcula
Precision@K, Recall@K y NDCG@K para varios K y los reúne en una tabla con una fila
por método. Es el artefacto central de la sección "Resultados" de la memoria.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k

RESULTS_CSV_PATH = Path("data/processed/results.csv")
DEFAULT_KS: tuple[int, ...] = (50, 100, 500)

logger = logging.getLogger(__name__)


def build_comparison_table(
    scores: dict[str, pd.Series],
    y_true: pd.Series,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> pd.DataFrame:
    """Construye la tabla de métricas para cada método.

    Args:
        scores: Mapa nombre_método -> Serie de scores (mayor = más prioritario),
            indexada igual que `y_true`.
        y_true: Serie binaria de ground truth.
        ks: Cortes K a reportar.

    Returns:
        DataFrame indexado por `method` con columnas `P@K`, `R@K`, `NDCG@K`.
    """
    y = pd.Series(y_true).astype(float)
    rows = []
    for name, raw in scores.items():
        s = pd.Series(raw).reindex(y.index)
        row: dict[str, float | str] = {"method": name}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(s.to_numpy(), y.to_numpy(), k)
            row[f"R@{k}"] = recall_at_k(s.to_numpy(), y.to_numpy(), k)
            row[f"NDCG@{k}"] = ndcg_at_k(s.to_numpy(), y.to_numpy(), k)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def save_results(df: pd.DataFrame, path: Path = RESULTS_CSV_PATH) -> Path:
    """Escribe la tabla comparativa en CSV. Devuelve la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("tabla de resultados escrita en %s", path)
    return path
