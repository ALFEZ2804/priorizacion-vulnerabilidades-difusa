"""Wrapper jerárquico del FIS: encadena los 3 sub-FIS y aplica el
post-procesado KEV.

Pipeline:
    exploitability + impact      → FIS-1 (Severidad) → severity
    epss_percentile + epss_delta → FIS-2 (Amenaza)   → threat
    severity + threat + log_cpe  → FIS-3 (Prioridad) → priority_raw
    priority_raw + KEV_boost     →                     priority (clip [0, 10])

KEV no entra como input difuso: es observación confirmada, no incertidumbre.
Modelarlo como antecedente fuzzy contradiría el paradigma. Se aplica como
post-procesado determinista `+2 con clip [0, 10]` sobre `priority_raw`, lo
que preserva el orden relativo entre CVEs ya en KEV y separa la información
observada de la estimada por el FIS.

La propagación de NaN viene gratis: las 29 filas sin EPSS reciben
`threat = NaN` → `priority_raw = NaN` → `NaN + 2 → NaN` y `.clip()` respeta
NaN. No hace falta guard explícito.
"""

from __future__ import annotations

import pandas as pd

from src.fis.priority import compute_priority_fis3
from src.fis.severity import compute_severity
from src.fis.threat import compute_threat

KEV_BOOST = 2.0
PRIORITY_MIN = 0.0
PRIORITY_MAX = 10.0


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el FIS jerárquico completo + post-procesado KEV a `df`.

    Añade cuatro columnas al final: ``severity``, ``threat``, ``priority_raw``
    y ``priority``. No muta el input: devuelve una copia.

    Requiere las columnas: ``exploitability_subscore``, ``impact_subscore``
    (FIS-1), ``epss_percentile``, ``epss_delta`` (FIS-2), ``log_cpe`` (FIS-3)
    e ``in_kev`` (post-procesado).

    El coste por fila es el de tres llamadas a `skfuzzy.control` encadenadas:
    aceptable para tests y muestras pequeñas. Sobre los 92k+ CVEs del
    unified el bucle se vuelve lento; una versión vectorizada sustituye los bucles
    fila-a-fila por una implementación Mamdani vectorizada.
    """
    out = compute_severity(df)
    out = compute_threat(out)
    out = compute_priority_fis3(out)
    boost = out["in_kev"].astype(int) * KEV_BOOST
    out["priority"] = (out["priority_raw"] + boost).clip(PRIORITY_MIN, PRIORITY_MAX)
    return out
