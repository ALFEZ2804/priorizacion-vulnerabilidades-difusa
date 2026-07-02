"""FIS-1 Severidad: MFs, reglas y aplicación al DataFrame.

Construye los dos antecedentes (`exploitability_subscore`,
`impact_subscore`) y el consecuente (`severity`) con los cortes
triangulares del diseño del subsistema, ensambla
las 9 reglas Mamdani de la matriz 3×3 y expone `compute_severity(df)`
para aplicar el FIS-1 sobre un DataFrame del unified.

**Matriz de reglas: opción B (min suavizado).**
Las cruzadas (exploitability low + impact high, y simétrica) suben a
`medium` en lugar de quedar en `low` como haría un min puro. Razón:
respeta literalmente la frase del diseño "las cruzadas inclinan al
lado alto cuando el impact lo justifica", aprovecha mejor el rango
[0, 10] del output (B reparte 3/3/3 las celdas vs 5/3/1 de A) y no
descarta a priori CVEs con perfil cruzado que pueden ser positivos KEV.
A queda documentada como alternativa.

T-norma AND = `min` (default de `skfuzzy.control` con `&`). T-conorma
implícita = `max`. Implicación = Mamdani. Defuzzificación = centroide
(default del consecuente sin especificar).

Convenciones de labels:
- variables: nombres idénticos a las columnas del unified, lo que
  permite `fis.infer(**row.to_dict())` sin renombrar.
- términos: `low` / `medium` / `high`.
"""

from __future__ import annotations

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.fis import FuzzyInference
from src.fis._universe import universe

# Paso del universo: 0,1 para inputs (NVD publica los subscores con
# 1 decimal: más fino es ruido); 0,01 para el output porque la
# defuzzificación por centroide integra sobre el universo del
# consecuente y un paso fino mejora la precisión numérica.
_INPUT_STEP = 0.1
_OUTPUT_STEP = 0.01


def build_exploitability_antecedent() -> ctrl.Antecedent:
    """Antecedente `exploitability_subscore` ∈ [0,1; 3,9]."""
    var = ctrl.Antecedent(universe(0.1, 3.9, _INPUT_STEP), "exploitability_subscore")
    var["low"] = fuzz.trimf(var.universe, [0.1, 0.1, 1.4])
    var["medium"] = fuzz.trimf(var.universe, [0.9, 2.0, 3.2])
    var["high"] = fuzz.trimf(var.universe, [2.5, 3.9, 3.9])
    return var


def build_impact_antecedent() -> ctrl.Antecedent:
    """Antecedente `impact_subscore` ∈ [0; 6]."""
    var = ctrl.Antecedent(universe(0.0, 6.0, _INPUT_STEP), "impact_subscore")
    var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 2.5])
    var["medium"] = fuzz.trimf(var.universe, [1.5, 3.5, 5.0])
    var["high"] = fuzz.trimf(var.universe, [4.0, 6.0, 6.0])
    return var


def build_severity_consequent() -> ctrl.Consequent:
    """Consecuente `severity` ∈ [0; 10]."""
    var = ctrl.Consequent(universe(0.0, 10.0, _OUTPUT_STEP), "severity")
    var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 4.0])
    var["medium"] = fuzz.trimf(var.universe, [3.0, 5.0, 7.0])
    var["high"] = fuzz.trimf(var.universe, [6.0, 10.0, 10.0])
    return var


def _build_severity_rules(
    exp: ctrl.Antecedent,
    imp: ctrl.Antecedent,
    sev: ctrl.Consequent,
) -> list[ctrl.Rule]:
    """Matriz 3×3 de la opción B (min suavizado).

    Las cruzadas (exp_low + imp_high, exp_high + imp_low) caen en
    `medium`, no en `low` como haría min puro. El resto de combinaciones
    coincide con min(exp_level, imp_level).
    """
    return [
        ctrl.Rule(exp["low"]    & imp["low"],    sev["low"]),
        ctrl.Rule(exp["low"]    & imp["medium"], sev["low"]),
        ctrl.Rule(exp["low"]    & imp["high"],   sev["medium"]),
        ctrl.Rule(exp["medium"] & imp["low"],    sev["low"]),
        ctrl.Rule(exp["medium"] & imp["medium"], sev["medium"]),
        ctrl.Rule(exp["medium"] & imp["high"],   sev["high"]),
        ctrl.Rule(exp["high"]   & imp["low"],    sev["medium"]),
        ctrl.Rule(exp["high"]   & imp["medium"], sev["high"]),
        ctrl.Rule(exp["high"]   & imp["high"],   sev["high"]),
    ]


def build_severity_fis() -> FuzzyInference:
    """Ensambla el FIS-1 completo (MFs + 9 reglas) listo para `.infer()`."""
    exp = build_exploitability_antecedent()
    imp = build_impact_antecedent()
    sev = build_severity_consequent()
    return FuzzyInference(
        antecedents=[exp, imp],
        consequent=sev,
        rules=_build_severity_rules(exp, imp, sev),
    )


def compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna `severity` aplicando FIS-1 a todo el DataFrame.

    Requiere las columnas `exploitability_subscore` e `impact_subscore`.
    No muta el input: devuelve una copia con la columna añadida al final.

    Usa `FuzzyInference.infer_batch` (Mamdani vectorizado): una
    sola llamada sobre las columnas en lugar de un bucle fila a fila.
    """
    fis = build_severity_fis()
    out = df.copy()
    out["severity"] = fis.infer_batch(
        exploitability_subscore=out["exploitability_subscore"].to_numpy(dtype=float),
        impact_subscore=out["impact_subscore"].to_numpy(dtype=float),
    )
    return out
