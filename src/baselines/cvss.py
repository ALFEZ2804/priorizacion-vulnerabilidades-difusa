"""Baseline CVSS puro.

Rankea las vulnerabilidades por su `cvss_v3_score`. Es el baseline de referencia
del TFG: la severidad técnica estática que CVSS asigna, sin ninguna señal de
probabilidad de explotación ni de contexto. El FIS busca mejorar justamente este
ranking.
"""

from __future__ import annotations

import pandas as pd

SCORE_COLUMN = "cvss_v3_score"


def cvss_score(df: pd.DataFrame) -> pd.Series:
    """Score de priorización por CVSS (mayor = más prioritario).

    Args:
        df: DataFrame con la columna `cvss_v3_score`.

    Returns:
        Serie float indexada como `df`.
    """
    return df[SCORE_COLUMN].astype(float)


def rank_by_cvss(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `df` ordenado por CVSS descendente, con la columna `cvss_rank_score`."""
    return df.assign(cvss_rank_score=cvss_score(df)).sort_values(
        "cvss_rank_score", ascending=False
    )
