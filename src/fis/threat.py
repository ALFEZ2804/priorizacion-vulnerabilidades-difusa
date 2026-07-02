"""FIS-2 Amenaza: MFs, reglas y aplicación al DataFrame.

Construye los dos antecedentes (`epss_percentile`, `epss_delta`) y el
consecuente (`threat`) con los cortes del diseño del subsistema, ensambla las 9
reglas Mamdani de la matriz 3×3 y expone `compute_threat(df)` para aplicar el
FIS-2 sobre un DataFrame del unified.

**Matriz de reglas: opción B ("el delta rescata").** La única celda no trivial es `percentile low + delta increasing`,
que sube a `medium` en lugar de quedar en `low`: captura vulnerabilidades con
EPSS aún bajo pero atención emergente (recién activadas por FIRST). El resto de
la matriz sigue el criterio "percentile high domina; con percentile medium el
delta decide".

Los cortes de las MFs de `epss_percentile` **no se fijan a mano**: se calibran
por búsqueda aleatoria en calibración (`scripts/optimize_fis.py`, fitness NDCG@100
sobre `y_train`, validada out-of-time en test); la config ganadora está en
`data/processed/optim/best_config.json`. `low` y `medium` son trapezoidales y
`high` es **triangular** (no satura): así se preserva la resolución de EPSS en la
cabeza del ranking en vez de aplanarla, que era la limitación de la MF inicial.
`epss_delta` y `threat` conservan sus MFs y cortes de dominio.

T-norma AND = `min`, T-conorma = `max`, implicación Mamdani, defuzzificación
por centroide: igual que FIS-1. El output `threat` comparte los mismos cortes
que `severity` (low=[0,0,4], medium=[3,5,7], high=[6,10,10]).

Convenciones de labels: nombres de variables idénticos a las columnas del
unified (`epss_percentile`, `epss_delta`) para poder `fis.infer(**row)` sin
renombrar.
"""

from __future__ import annotations

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.fis import FuzzyInference
from src.fis._universe import universe

# Pasos más finos que en FIS-1 (0,1): aquí los dominios son estrechos
# ([0,1] y [-1,1]) y los cortes caen en décimas/centésimas, así que 0,01
# da resolución suficiente sin inflar el coste. El output usa 0,01 por la
# misma razón que en severity (precisión del centroide).
_PERCENTILE_STEP = 0.01
_DELTA_STEP = 0.01
_OUTPUT_STEP = 0.01


# Cortes a<b<c<d de epss_percentile calibrados por búsqueda aleatoria en
# calibración (scripts/optimize_fis.py → data/processed/optim/best_config.json).
# Se conservan con su precisión completa para reproducir exactamente los
# resultados validados; redondearlos desplazaría las métricas.
_PCT_A, _PCT_B, _PCT_C, _PCT_D = (
    0.6981207839786372,
    0.7721083991043806,
    0.8379802186553831,
    0.9277954301722804,
)


def build_epss_percentile_antecedent() -> ctrl.Antecedent:
    """Antecedente `epss_percentile` ∈ [0; 1] con cortes calibrados (ver módulo).

    `low`/`medium` trapezoidales, `high` triangular (no satura). Los cortes
    provienen de la calibración por búsqueda, no de valores puestos a mano.
    """
    var = ctrl.Antecedent(universe(0.0, 1.0, _PERCENTILE_STEP), "epss_percentile")
    var["low"] = fuzz.trapmf(var.universe, [0.0, 0.0, _PCT_A, _PCT_B])
    var["medium"] = fuzz.trapmf(var.universe, [_PCT_A, _PCT_B, _PCT_C, _PCT_D])
    var["high"] = fuzz.trimf(var.universe, [_PCT_D, 1.0, 1.0])
    return var


def build_epss_delta_antecedent() -> ctrl.Antecedent:
    """Antecedente `epss_delta` ∈ [-1; 1] con MFs triangulares centradas en 0."""
    var = ctrl.Antecedent(universe(-1.0, 1.0, _DELTA_STEP), "epss_delta")
    var["decreasing"] = fuzz.trimf(var.universe, [-1.0, -1.0, -0.1])
    var["stable"] = fuzz.trimf(var.universe, [-0.2, 0.0, 0.2])
    var["increasing"] = fuzz.trimf(var.universe, [0.1, 1.0, 1.0])
    return var


def build_threat_consequent() -> ctrl.Consequent:
    """Consecuente `threat` ∈ [0; 10] (mismos cortes que `severity`)."""
    var = ctrl.Consequent(universe(0.0, 10.0, _OUTPUT_STEP), "threat")
    var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 4.0])
    var["medium"] = fuzz.trimf(var.universe, [3.0, 5.0, 7.0])
    var["high"] = fuzz.trimf(var.universe, [6.0, 10.0, 10.0])
    return var


def _build_threat_rules(
    pct: ctrl.Antecedent,
    dlt: ctrl.Antecedent,
    thr: ctrl.Consequent,
) -> list[ctrl.Rule]:
    """Matriz 3×3: opción B.

    | percentile \\ delta | decreasing | stable | increasing |
    |---------------------|:----------:|:------:|:----------:|
    | low                 | low        | low    | medium     |
    | medium              | low        | medium | high       |
    | high                | high       | high   | high       |
    """
    return [
        ctrl.Rule(pct["low"]    & dlt["decreasing"], thr["low"]),
        ctrl.Rule(pct["low"]    & dlt["stable"],     thr["low"]),
        ctrl.Rule(pct["low"]    & dlt["increasing"], thr["medium"]),
        ctrl.Rule(pct["medium"] & dlt["decreasing"], thr["low"]),
        ctrl.Rule(pct["medium"] & dlt["stable"],     thr["medium"]),
        ctrl.Rule(pct["medium"] & dlt["increasing"], thr["high"]),
        ctrl.Rule(pct["high"]   & dlt["decreasing"], thr["high"]),
        ctrl.Rule(pct["high"]   & dlt["stable"],     thr["high"]),
        ctrl.Rule(pct["high"]   & dlt["increasing"], thr["high"]),
    ]


def build_threat_fis() -> FuzzyInference:
    """Ensambla el FIS-2 completo (MFs + 9 reglas) listo para `.infer()`."""
    pct = build_epss_percentile_antecedent()
    dlt = build_epss_delta_antecedent()
    thr = build_threat_consequent()
    return FuzzyInference(
        antecedents=[pct, dlt],
        consequent=thr,
        rules=_build_threat_rules(pct, dlt, thr),
    )


def compute_threat(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna `threat` aplicando FIS-2 a todo el DataFrame.

    Requiere las columnas `epss_percentile` y `epss_delta`. No muta el input:
    devuelve una copia con la columna añadida al final.

    A diferencia de `compute_severity` (inputs con 100 % de cobertura),
    `epss_percentile`/`epss_delta` tienen 29 filas sin EPSS (cobertura
    99,97 %). `infer_batch` propaga esos NaN sin esfuerzo: una fila con algún
    input NaN devuelve `threat = NaN`. El tratamiento definitivo de esos
    missings se aborda en el análisis de robustez.
    """
    fis = build_threat_fis()
    out = df.copy()
    out["threat"] = fis.infer_batch(
        epss_percentile=out["epss_percentile"].to_numpy(dtype=float),
        epss_delta=out["epss_delta"].to_numpy(dtype=float),
    )
    return out
