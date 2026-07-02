"""Baseline combinado simple CVSS × EPSS.

Baseline más fuerte que CVSS solo: aproxima riesgo = severidad × probabilidad de
explotación. Se normaliza `cvss_v3_score` a [0, 1] (división por 10) y se multiplica
por `epss_percentile`. Es la forma más directa de combinar ambas señales sin lógica
difusa; sirve de referencia para aislar el valor que aporta el FIS frente a una
combinación lineal trivial de las mismas dos fuentes.

Se elige el **producto** (no el promedio) porque modela la conjunción "grave Y
probable": un CVE grave pero improbable, o probable pero leve, no debe encabezar el
ranking. El promedio los premiaría en exceso.
"""

from __future__ import annotations

import pandas as pd

CVSS_COLUMN = "cvss_v3_score"
EPSS_COLUMN = "epss_percentile"
CVSS_MAX = 10.0


def combined_score(df: pd.DataFrame) -> pd.Series:
    """Score combinado (cvss_v3_score / 10) * epss_percentile.

    Args:
        df: DataFrame con `cvss_v3_score` y `epss_percentile`.

    Returns:
        Serie float indexada como `df` (NaN donde falte EPSS).
    """
    cvss_norm = df[CVSS_COLUMN].astype(float) / CVSS_MAX
    return cvss_norm * df[EPSS_COLUMN].astype(float)


def rank_by_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve `df` ordenado por el score combinado descendente."""
    return df.assign(combined_rank_score=combined_score(df)).sort_values(
        "combined_rank_score", ascending=False
    )
