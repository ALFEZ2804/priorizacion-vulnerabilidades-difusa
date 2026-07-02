"""Smoke tests del FIS-3 Prioridad.

12.1: MFs (severity/threat heredadas, log_cpe triangular, priority_raw 4 niveles),
breakpoints contra el diseño, cobertura sin huecos.
12.2: las 27 reglas verificadas contra la fórmula `clip(max(sev,thr)+mod)`,
reproducción del ejemplo de diseño y `compute_priority_fis3` sobre DataFrame.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from src.fis.priority import (
    build_log_cpe_antecedent,
    build_priority_consequent,
    build_priority_fis,
    build_severity_antecedent,
    build_threat_antecedent,
    compute_priority_fis3,
)


def test_antecedents_have_expected_labels():
    assert build_severity_antecedent().label == "severity"
    assert build_threat_antecedent().label == "threat"
    assert build_log_cpe_antecedent().label == "log_cpe"


def test_consequent_has_four_levels():
    pri = build_priority_consequent()
    assert pri.label == "priority_raw"
    assert set(pri.terms.keys()) == {"low", "medium", "high", "critical"}


def test_terms_match_design():
    assert set(build_severity_antecedent().terms.keys()) == {"low", "medium", "high"}
    assert set(build_threat_antecedent().terms.keys()) == {"low", "medium", "high"}
    assert set(build_log_cpe_antecedent().terms.keys()) == {"narrow", "medium", "wide"}


def test_mf_breakpoints_match_design_doc():
    """Contrato con el diseño del subsistema (todas triangulares)."""
    sev = build_severity_antecedent()
    cpe = build_log_cpe_antecedent()
    pri = build_priority_consequent()

    cases = [
        (sev, "low", [0.0, 0.0, 4.0]),
        (sev, "medium", [3.0, 5.0, 7.0]),
        (sev, "high", [6.0, 10.0, 10.0]),
        (cpe, "narrow", [0.69, 0.69, 2.0]),
        (cpe, "medium", [1.0, 2.5, 4.5]),
        (cpe, "wide", [3.0, 7.97, 7.97]),
        (pri, "low", [0.0, 0.0, 3.0]),
        (pri, "medium", [2.0, 4.0, 6.0]),
        (pri, "high", [5.0, 6.5, 8.0]),
        (pri, "critical", [7.0, 10.0, 10.0]),
    ]
    for var, label, bps in cases:
        expected = fuzz.trimf(var.universe, bps)
        assert np.allclose(var.terms[label].mf, expected), (
            f"{var.label}[{label}] no coincide con triángulo {bps}"
        )


def test_mf_coverage_no_gaps():
    variables = [
        (build_severity_antecedent(), ("low", "medium", "high")),
        (build_log_cpe_antecedent(), ("narrow", "medium", "wide")),
        (build_priority_consequent(), ("low", "medium", "high", "critical")),
    ]
    for var, labels in variables:
        lo, hi = float(var.universe[0]), float(var.universe[-1])
        points = np.linspace(lo, hi, 1000)
        memberships = np.stack(
            [
                fuzz.interp_membership(var.universe, var.terms[label].mf, points)
                for label in labels
            ]
        )
        gap_mask = memberships.max(axis=0) <= 0
        assert not gap_mask.any(), (
            f"{var.label}: hueco de cobertura en x={points[gap_mask][0]:.3f}"
        )


# --- -----------------------------------------------------------


def test_build_priority_fis_constructs():
    fis = build_priority_fis()
    assert fis.antecedent_names == ["severity", "threat", "log_cpe"]
    assert fis.consequent_name == "priority_raw"


# Valores en el pico de cada MF: activan exactamente un nivel con μ=1, de modo
# que sólo una de las 27 reglas dispara (peso 1) y el output es el centroide de
# su consecuente.
_LEVEL_PEAK = {"low": 0.0, "medium": 5.0, "high": 10.0}
_CPE_PEAK = {"narrow": 0.69, "medium": 2.5, "wide": 7.97}
_LEVEL_IDX = {"low": 0, "medium": 1, "high": 2}
# Centroides esperados de cada nivel de priority_raw clipado a 1.0.
_PRIORITY_INDEX_BANDS = [2.5, 5.25, 7.75]  # fronteras low|medium|high|critical


def _priority_band(value: float) -> int:
    """Mapea un priority_raw crisp al índice de nivel (0=low … 3=critical)."""
    for idx, frontier in enumerate(_PRIORITY_INDEX_BANDS):
        if value < frontier:
            return idx
    return 3


def test_rules_follow_threat_leads():
    """Las 27 reglas == clip(threat_level + (sev==high) + (cpe==wide), low, critical).

    Verificación funcional de la recalibración: `threat` lidera y
    `severity`/`log_cpe` solo agravan. Para cada combinación de niveles se inyectan
    los picos de las MFs, se infiere priority_raw y se comprueba que cae en la banda
    del nivel que predice la fórmula.
    """
    fis = build_priority_fis()
    for sev_lvl, thr_lvl, cpe_lvl in itertools.product(
        ("low", "medium", "high"), ("low", "medium", "high"), ("narrow", "medium", "wide")
    ):
        expected = min(
            _LEVEL_IDX[thr_lvl]
            + (1 if sev_lvl == "high" else 0)
            + (1 if cpe_lvl == "wide" else 0),
            3,
        )
        pr = fis.infer(
            severity=_LEVEL_PEAK[sev_lvl],
            threat=_LEVEL_PEAK[thr_lvl],
            log_cpe=_CPE_PEAK[cpe_lvl],
        )
        assert math.isfinite(pr)
        assert _priority_band(pr) == expected, (
            f"sev={sev_lvl}, threat={thr_lvl}, log_cpe={cpe_lvl}: "
            f"priority_raw={pr:.3f} (banda {_priority_band(pr)}) ≠ esperado {expected}"
        )


def test_threat_leads_key_cases():
    """Comportamiento clave de la recalibración: `threat` lidera."""
    fis = build_priority_fis()
    # threat alto + severity alta -> critical
    assert _priority_band(fis.infer(severity=10.0, threat=10.0, log_cpe=0.69)) == 3
    # severity alta SIN amenaza -> solo medium (la severidad por sí sola no encabeza)
    assert _priority_band(fis.infer(severity=10.0, threat=0.0, log_cpe=0.69)) == 1
    # threat alto domina aunque la severidad sea baja -> high
    assert _priority_band(fis.infer(severity=0.0, threat=10.0, log_cpe=0.69)) == 2


def test_compute_priority_fis3_adds_column_without_mutation():
    df_in = pd.DataFrame(
        {
            "severity": [1.0, 5.0, 9.0],
            "threat": [1.0, 5.0, 9.0],
            "log_cpe": [0.69, 2.5, 7.0],
        }
    )
    df_out = compute_priority_fis3(df_in)

    assert "priority_raw" not in df_in.columns
    assert "priority_raw" in df_out.columns
    assert df_out["priority_raw"].apply(math.isfinite).all()

    pr = df_out["priority_raw"].tolist()
    assert pr[0] < pr[1] < pr[2], f"esperada monotonía, obtenido {pr}"


def test_compute_priority_fis3_propagates_nan():
    """Filas con severity/threat/log_cpe NaN reciben priority_raw=NaN."""
    df_in = pd.DataFrame(
        {
            "severity": [6.0, np.nan, 6.0],
            "threat": [5.0, 5.0, np.nan],
            "log_cpe": [3.0, 3.0, 3.0],
        }
    )
    df_out = compute_priority_fis3(df_in)
    assert math.isfinite(df_out["priority_raw"].iloc[0])
    assert math.isnan(df_out["priority_raw"].iloc[1])
    assert math.isnan(df_out["priority_raw"].iloc[2])
