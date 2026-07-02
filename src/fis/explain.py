"""Explicación regla a regla de la prioridad de una vulnerabilidad (XAI).

Para un CVE concreto, reconstruye de forma **transparente** el razonamiento de los
tres sub-FIS: el grado de pertenencia de cada entrada a sus etiquetas, la fuerza
con la que se activa cada regla (el mínimo de sus antecedentes, la T-norma de
Mamdani) y el nivel de salida dominante. Termina con una explicación en lenguaje
natural de por qué la vulnerabilidad recibió su prioridad.

No se usan detalles internos de `skfuzzy`: el cálculo se rehace a partir de las
funciones de pertenencia y las matrices de reglas de `severity.py`, `threat.py` y
`priority.py` (las mismas que documenta la memoria), de modo que cada paso es el
Mamdani de manual y es defendible línea a línea. El valor numérico defuzzificado se
toma del propio motor (`FuzzyInference`), así que coincide con `scored.parquet`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz

from src.fis.severity import (
    build_exploitability_antecedent,
    build_impact_antecedent,
    build_severity_fis,
)
from src.fis.threat import (
    build_epss_delta_antecedent,
    build_epss_percentile_antecedent,
    build_threat_fis,
)
from src.fis.priority import (
    build_log_cpe_antecedent,
    build_severity_antecedent,
    build_threat_antecedent,
    build_priority_fis,
)

# --- Matrices de reglas (idénticas a las de los módulos del FIS) ------------- #
# Se reproducen aquí como datos para poder explicarlas; un test cruza esta tabla
# con las reglas reales para garantizar que no se desincronizan.
SEVERITY_RULES: dict[tuple[str, str], str] = {
    ("low", "low"): "low", ("low", "medium"): "low", ("low", "high"): "medium",
    ("medium", "low"): "low", ("medium", "medium"): "medium", ("medium", "high"): "high",
    ("high", "low"): "medium", ("high", "medium"): "high", ("high", "high"): "high",
}
THREAT_RULES: dict[tuple[str, str], str] = {
    ("low", "decreasing"): "low", ("low", "stable"): "low", ("low", "increasing"): "medium",
    ("medium", "decreasing"): "low", ("medium", "stable"): "medium", ("medium", "increasing"): "high",
    ("high", "decreasing"): "high", ("high", "stable"): "high", ("high", "increasing"): "high",
}


def _priority_rules() -> dict[tuple[str, str, str], str]:
    """27 reglas: priority = clip(threat + (sev==high) + (cpe==wide), low, critical)."""
    levels = ("low", "medium", "high")
    out = ("low", "medium", "high", "critical")
    rank = {"low": 0, "medium": 1, "high": 2}
    rules: dict[tuple[str, str, str], str] = {}
    for s in levels:
        for t in levels:
            base = rank[t]
            for c, mod in (("narrow", 0), ("medium", 0), ("wide", 1)):
                level = max(0, min(3, base + (1 if s == "high" else 0) + mod))
                rules[(s, t, c)] = out[level]
    return rules


PRIORITY_RULES = _priority_rules()

# Traducciones de etiquetas para la explicación en lenguaje natural.
_ES = {
    "low": "baja", "medium": "media", "high": "alta", "critical": "crítica",
    "decreasing": "decreciente", "stable": "estable", "increasing": "creciente",
    "narrow": "estrecha", "wide": "amplia",
}

# Motores cacheados (se construyen una vez): aportan el valor defuzzificado.
_SEV_FIS = build_severity_fis()
_THR_FIS = build_threat_fis()
_PRI_FIS = build_priority_fis()


@dataclass
class FiredRule:
    """Una regla activada: etiquetas de sus antecedentes, salida y fuerza."""
    antecedents: tuple[str, ...]
    output: str
    strength: float


@dataclass
class SubFISExplanation:
    """Explicación de un sub-FIS: pertenencias, reglas activadas y salida."""
    memberships: dict[str, dict[str, float]]   # variable -> {etiqueta: grado}
    fired_rules: list[FiredRule]               # reglas con fuerza > 0, desc.
    output_value: float
    dominant_label: str                        # etiqueta de salida más fuerte


def _memberships(antecedent, value: float) -> dict[str, float]:
    """Grado de pertenencia de `value` a cada etiqueta del antecedente."""
    return {
        label: round(float(fuzz.interp_membership(antecedent.universe, antecedent[label].mf, value)), 4)
        for label in antecedent.terms
    }


def _fired(rules: dict[tuple, str], degrees: list[dict[str, float]],
           min_strength: float = 1e-6) -> list[FiredRule]:
    """Fuerza de cada regla = min de los grados de sus antecedentes (Mamdani)."""
    fired: list[FiredRule] = []
    for antecedents, output in rules.items():
        strength = min(deg.get(lbl, 0.0) for deg, lbl in zip(degrees, antecedents))
        if strength > min_strength:
            fired.append(FiredRule(antecedents, output, round(strength, 4)))
    fired.sort(key=lambda r: r.strength, reverse=True)
    return fired


def _dominant(fired: list[FiredRule]) -> str:
    """Etiqueta de salida con mayor activación agregada (T-conorma max)."""
    agg: dict[str, float] = {}
    for r in fired:
        agg[r.output] = max(agg.get(r.output, 0.0), r.strength)
    return max(agg, key=agg.get) if agg else "low"


def explain_severity(exploitability: float, impact: float) -> SubFISExplanation:
    """FIS-1: explica la severidad a partir de los dos subíndices CVSS."""
    exp = build_exploitability_antecedent()
    imp = build_impact_antecedent()
    deg_e = _memberships(exp, exploitability)
    deg_i = _memberships(imp, impact)
    fired = _fired(SEVERITY_RULES, [deg_e, deg_i])
    value = _SEV_FIS.infer(exploitability_subscore=exploitability, impact_subscore=impact)
    return SubFISExplanation(
        {"exploitability_subscore": deg_e, "impact_subscore": deg_i},
        fired, round(float(value), 2), _dominant(fired),
    )


def explain_threat(epss_percentile: float, epss_delta: float) -> SubFISExplanation:
    """FIS-2: explica la amenaza a partir del percentil de EPSS y su variación."""
    pct = build_epss_percentile_antecedent()
    dlt = build_epss_delta_antecedent()
    deg_p = _memberships(pct, epss_percentile)
    deg_d = _memberships(dlt, epss_delta)
    fired = _fired(THREAT_RULES, [deg_p, deg_d])
    value = _THR_FIS.infer(epss_percentile=epss_percentile, epss_delta=epss_delta)
    return SubFISExplanation(
        {"epss_percentile": deg_p, "epss_delta": deg_d},
        fired, round(float(value), 2), _dominant(fired),
    )


def explain_priority(severity: float, threat: float, log_cpe: float) -> SubFISExplanation:
    """FIS-3: explica la prioridad bruta a partir de severidad, amenaza y superficie."""
    sev = build_severity_antecedent()
    thr = build_threat_antecedent()
    cpe = build_log_cpe_antecedent()
    deg_s = _memberships(sev, severity)
    deg_t = _memberships(thr, threat)
    deg_c = _memberships(cpe, log_cpe)
    fired = _fired(PRIORITY_RULES, [deg_s, deg_t, deg_c])
    value = _PRI_FIS.infer(severity=severity, threat=threat, log_cpe=log_cpe)
    return SubFISExplanation(
        {"severity": deg_s, "threat": deg_t, "log_cpe": deg_c},
        fired, round(float(value), 2), _dominant(fired),
    )


def explain_cve(*, exploitability_subscore: float, impact_subscore: float,
                epss_percentile: float, epss_delta: float, log_cpe: float,
                in_kev: bool) -> dict:
    """Explicación completa de la prioridad de un CVE (cadena de los 3 sub-FIS).

    Devuelve un diccionario con la explicación de cada sub-FIS, el ajuste por KEV
    y un resumen en lenguaje natural, todo serializable para la web.
    """
    sev = explain_severity(exploitability_subscore, impact_subscore)
    thr = explain_threat(epss_percentile, epss_delta)
    pri = explain_priority(sev.output_value, thr.output_value, log_cpe)
    priority = min(10.0, pri.output_value + (2.0 if in_kev else 0.0))

    summary = _build_summary(sev, thr, pri, in_kev, priority)
    return {
        "severity": _subfis_dict(sev),
        "threat": _subfis_dict(thr),
        "priority_raw": _subfis_dict(pri),
        "kev_boost": 2.0 if in_kev else 0.0,
        "priority": round(priority, 2),
        "summary": summary,
    }


def _subfis_dict(e: SubFISExplanation) -> dict:
    return {
        "value": e.output_value,
        "dominant_label": e.dominant_label,
        "dominant_label_es": _ES.get(e.dominant_label, e.dominant_label),
        "memberships": e.memberships,
        "fired_rules": [
            {"antecedents": r.antecedents, "output": r.output, "strength": r.strength}
            for r in e.fired_rules
        ],
    }


def _build_summary(sev, thr, pri, in_kev: bool, priority: float) -> str:
    """Frase explicativa: por qué este CVE tiene esta prioridad."""
    nivel = _ES.get(pri.dominant_label, pri.dominant_label)
    partes = [
        f"Prioridad {priority:.1f} ({nivel}).",
        f"La severidad técnica es {_ES[sev.dominant_label]} ({sev.output_value:.1f}) "
        f"y la amenaza es {_ES[thr.dominant_label]} ({thr.output_value:.1f}).",
    ]
    if pri.fired_rules:
        top = pri.fired_rules[0]
        s_lbl, t_lbl, c_lbl = top.antecedents
        partes.append(
            f"La regla dominante combina severidad {_ES[s_lbl]} y amenaza {_ES[t_lbl]} "
            f"con una superficie afectada {_ES[c_lbl]}, y concluye prioridad {_ES[top.output]}."
        )
    if in_kev:
        partes.append("Además, está confirmada como explotada en el catálogo KEV, "
                      "lo que añade 2 puntos a su prioridad.")
    return " ".join(partes)
