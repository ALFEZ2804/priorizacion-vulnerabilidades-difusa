"""FIS-3 Prioridad: MFs, 27 reglas y aplicación al DataFrame.

Sub-FIS terminal del modelo jerárquico. Toma `severity` (salida de FIS-1),
`threat` (salida de FIS-2) y `log_cpe`, y produce `priority_raw ∈ [0, 10]`.
Los cortes y las reglas siguen el diseño del subsistema de prioridad.

`severity` y `threat` entran como antecedentes con las **mismas** MFs que tenían
como consecuentes en FIS-1/FIS-2 (low/medium/high = (0,0,4)/(3,5,7)/(6,10,10)).
Se redefinen aquí porque en `skfuzzy.control` un `Antecedent` y un `Consequent`
son tipos distintos y no se pueden reutilizar las instancias de los otros módulos.

**27 reglas: `threat` lidera; `severity` y `log_cpe` solo agravan (recalibrado).**
La prioridad la marca la **probabilidad de explotación** (`threat`, derivado de EPSS),
que es la señal que predice la entrada en KEV. `severity` y `log_cpe` son agravantes
que solo suman: una severidad alta (`high`) o una superficie amplia (`wide`) suben un
nivel; nada rebaja por debajo del nivel de `threat`. Así, un CVE grave pero sin
amenaza no encabeza el ranking, y uno con amenaza alta sube aunque su severidad o
superficie sean modestas. Formalmente:

    base = threat_level                                ∈ {low(0), medium(1), high(2)}
    mod  = (severity == high) + (log_cpe == wide)      ∈ {0, +1, +2}
    priority_level = clip(base + mod, low(0), critical(3))

El output usa **4 niveles** (low/medium/high/critical) para dar más resolución a
la cola alta del ranking. `priority_raw` es la salida bruta; el post-procesado
KEV (+2) se aplica en el wrapper jerárquico, no aquí.

**Nota sobre `log_cpe`:** su distribución está muy concentrada: el P75
coincide con el mínimo (0.693 = log1p(1)): >75 % de los CVEs afectan a una sola
configuración CPE, 95,5 % caen en `narrow` y solo 2,3 % en `wide`. El diseño
original (`narrow = -1`) rebajaba un nivel a casi todo el dataset y hundía a la
banda media incluso CVEs con `severity` y `threat` altos, dejándolos fuera de la
cola alta (FIS a 0 en P@K frente a EPSS). La recalibración hace
`narrow` identidad: `log_cpe` solo discrimina ya el caso `wide` (+1), coherente con
su escasa capacidad discriminante.

T-norma AND = `min`, T-conorma = `max`, implicación Mamdani, defuzzificación por
centroide: igual que FIS-1 y FIS-2.
"""

from __future__ import annotations

import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.fis import FuzzyInference
from src.fis._universe import universe

# severity/threat: cortes enteros sobre [0,10], 0,1 basta. log_cpe: dominio
# estrecho con extremos no enteros → 0,01. Output: 0,01 por precisión del
# centroide (mismo criterio que en severity/threat).
_LEVEL_STEP = 0.1
_LOGCPE_STEP = 0.01
_OUTPUT_STEP = 0.01

# Dominio observado de log_cpe en el unified (= [log1p(1), log1p(2895)]).
_LOGCPE_MIN = 0.69
_LOGCPE_MAX = 7.97


def _build_level_antecedent(label: str) -> ctrl.Antecedent:
    """Antecedente ∈ [0,10] con las MFs heredadas del output de FIS-1/FIS-2."""
    var = ctrl.Antecedent(universe(0.0, 10.0, _LEVEL_STEP), label)
    var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 4.0])
    var["medium"] = fuzz.trimf(var.universe, [3.0, 5.0, 7.0])
    var["high"] = fuzz.trimf(var.universe, [6.0, 10.0, 10.0])
    return var


def build_severity_antecedent() -> ctrl.Antecedent:
    """Antecedente `severity` ∈ [0,10] (MFs heredadas del consecuente de FIS-1)."""
    return _build_level_antecedent("severity")


def build_threat_antecedent() -> ctrl.Antecedent:
    """Antecedente `threat` ∈ [0,10] (MFs heredadas del consecuente de FIS-2)."""
    return _build_level_antecedent("threat")


def build_log_cpe_antecedent() -> ctrl.Antecedent:
    """Antecedente `log_cpe` ∈ [0,69; 7,97] con MFs triangulares narrow/medium/wide."""
    var = ctrl.Antecedent(universe(_LOGCPE_MIN, _LOGCPE_MAX, _LOGCPE_STEP), "log_cpe")
    var["narrow"] = fuzz.trimf(var.universe, [_LOGCPE_MIN, _LOGCPE_MIN, 2.0])
    var["medium"] = fuzz.trimf(var.universe, [1.0, 2.5, 4.5])
    var["wide"] = fuzz.trimf(var.universe, [3.0, _LOGCPE_MAX, _LOGCPE_MAX])
    return var


def build_priority_consequent() -> ctrl.Consequent:
    """Consecuente `priority_raw` ∈ [0,10] con 4 niveles."""
    var = ctrl.Consequent(universe(0.0, 10.0, _OUTPUT_STEP), "priority_raw")
    var["low"] = fuzz.trimf(var.universe, [0.0, 0.0, 3.0])
    var["medium"] = fuzz.trimf(var.universe, [2.0, 4.0, 6.0])
    var["high"] = fuzz.trimf(var.universe, [5.0, 6.5, 8.0])
    var["critical"] = fuzz.trimf(var.universe, [7.0, 10.0, 10.0])
    return var


def _build_priority_rules(
    sev: ctrl.Antecedent,
    thr: ctrl.Antecedent,
    cpe: ctrl.Antecedent,
    pri: ctrl.Consequent,
) -> list[ctrl.Rule]:
    """27 reglas (3×3×3), una matriz por nivel de `log_cpe` (recalibrado).

    Cada celda = `clip(threat_level + (sev==high) + (cpe==wide), low, critical)`.
    `test_fis_priority.py` verifica esa equivalencia regla a regla.
    """
    return [
        # priority = clip(threat_nivel + (sev high: +1) + (log_cpe wide: +1), low, critical)
        # --- log_cpe = narrow (sin agravante de superficie) ---
        ctrl.Rule(sev["low"]    & thr["low"]    & cpe["narrow"], pri["low"]),
        ctrl.Rule(sev["low"]    & thr["medium"] & cpe["narrow"], pri["medium"]),
        ctrl.Rule(sev["low"]    & thr["high"]   & cpe["narrow"], pri["high"]),
        ctrl.Rule(sev["medium"] & thr["low"]    & cpe["narrow"], pri["low"]),
        ctrl.Rule(sev["medium"] & thr["medium"] & cpe["narrow"], pri["medium"]),
        ctrl.Rule(sev["medium"] & thr["high"]   & cpe["narrow"], pri["high"]),
        ctrl.Rule(sev["high"]   & thr["low"]    & cpe["narrow"], pri["medium"]),
        ctrl.Rule(sev["high"]   & thr["medium"] & cpe["narrow"], pri["high"]),
        ctrl.Rule(sev["high"]   & thr["high"]   & cpe["narrow"], pri["critical"]),
        # --- log_cpe = medium (sin agravante de superficie, = narrow) ---
        ctrl.Rule(sev["low"]    & thr["low"]    & cpe["medium"], pri["low"]),
        ctrl.Rule(sev["low"]    & thr["medium"] & cpe["medium"], pri["medium"]),
        ctrl.Rule(sev["low"]    & thr["high"]   & cpe["medium"], pri["high"]),
        ctrl.Rule(sev["medium"] & thr["low"]    & cpe["medium"], pri["low"]),
        ctrl.Rule(sev["medium"] & thr["medium"] & cpe["medium"], pri["medium"]),
        ctrl.Rule(sev["medium"] & thr["high"]   & cpe["medium"], pri["high"]),
        ctrl.Rule(sev["high"]   & thr["low"]    & cpe["medium"], pri["medium"]),
        ctrl.Rule(sev["high"]   & thr["medium"] & cpe["medium"], pri["high"]),
        ctrl.Rule(sev["high"]   & thr["high"]   & cpe["medium"], pri["critical"]),
        # --- log_cpe = wide (agravante de superficie: +1) ---
        ctrl.Rule(sev["low"]    & thr["low"]    & cpe["wide"], pri["medium"]),
        ctrl.Rule(sev["low"]    & thr["medium"] & cpe["wide"], pri["high"]),
        ctrl.Rule(sev["low"]    & thr["high"]   & cpe["wide"], pri["critical"]),
        ctrl.Rule(sev["medium"] & thr["low"]    & cpe["wide"], pri["medium"]),
        ctrl.Rule(sev["medium"] & thr["medium"] & cpe["wide"], pri["high"]),
        ctrl.Rule(sev["medium"] & thr["high"]   & cpe["wide"], pri["critical"]),
        ctrl.Rule(sev["high"]   & thr["low"]    & cpe["wide"], pri["high"]),
        ctrl.Rule(sev["high"]   & thr["medium"] & cpe["wide"], pri["critical"]),
        ctrl.Rule(sev["high"]   & thr["high"]   & cpe["wide"], pri["critical"]),
    ]


def build_priority_fis() -> FuzzyInference:
    """Ensambla el FIS-3 completo (MFs + 27 reglas) listo para `.infer()`."""
    sev = build_severity_antecedent()
    thr = build_threat_antecedent()
    cpe = build_log_cpe_antecedent()
    pri = build_priority_consequent()
    return FuzzyInference(
        antecedents=[sev, thr, cpe],
        consequent=pri,
        rules=_build_priority_rules(sev, thr, cpe, pri),
    )


def compute_priority_fis3(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna `priority_raw` aplicando FIS-3 a todo el DataFrame.

    Requiere las columnas `severity`, `threat` (salidas de FIS-1 y FIS-2, que el
    wrapper jerárquico produce con `compute_severity` y `compute_threat`) y
    `log_cpe`. No muta el input.

    Como `threat` arrastra NaN en las 29 filas sin EPSS, `infer_batch` propaga
    ese NaN a `priority_raw` en esas filas.
    """
    fis = build_priority_fis()
    out = df.copy()
    out["priority_raw"] = fis.infer_batch(
        severity=out["severity"].to_numpy(dtype=float),
        threat=out["threat"].to_numpy(dtype=float),
        log_cpe=out["log_cpe"].to_numpy(dtype=float),
    )
    return out
