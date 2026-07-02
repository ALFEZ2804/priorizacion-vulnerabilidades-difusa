"""Baseline EPSS.

Rankea por `epss_percentile`, el percentil de la probabilidad de explotación a 30
días estimada por EPSS. Es el baseline más competitivo en captura de KEV porque
modela likelihood de explotación, justo lo que CVSS ignora. El TFG lo usa tanto
como baseline como señal de entrada del FIS (sub-FIS de Amenaza).
"""

from __future__ import annotations

import pandas as pd

SCORE_COLUMN = "epss_percentile"


def epss_score(df: pd.DataFrame) -> pd.Series:
    """Score de priorización por EPSS (mayor = más prioritario).

    Args:
        df: DataFrame con la columna `epss_percentile`.

    Returns:
        Serie float indexada como `df` (los NaN, filas sin EPSS, se conservan y
        las métricas los envían al fondo del ranking).
    """
    return df[SCORE_COLUMN].astype(float)


def rank_by_epss(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `df` ordenado por EPSS percentile descendente."""
    return df.assign(epss_rank_score=epss_score(df)).sort_values(
        "epss_rank_score", ascending=False
    )
