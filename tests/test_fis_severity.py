"""Smoke tests del FIS-1 Severidad.

MFs, breakpoints contra el diseño, cobertura sin huecos.
10.3: ensamblado del FIS, sanity Log4Shell, caso cruzado de la matriz B,
`compute_severity` sobre DataFrame.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from src.fis.severity import (
    build_exploitability_antecedent,
    build_impact_antecedent,
    build_severity_consequent,
    build_severity_fis,
    compute_severity,
)


def test_antecedents_have_expected_labels():
    exp = build_exploitability_antecedent()
    imp = build_impact_antecedent()
    assert exp.label == "exploitability_subscore"
    assert imp.label == "impact_subscore"


def test_consequent_has_expected_label():
    sev = build_severity_consequent()
    assert sev.label == "severity"


def test_each_variable_has_three_terms_named_low_medium_high():
    expected = {"low", "medium", "high"}
    for var in (
        build_exploitability_antecedent(),
        build_impact_antecedent(),
        build_severity_consequent(),
    ):
        assert set(var.terms.keys()) == expected, (
            f"{var.label} tiene términos {set(var.terms.keys())}, "
            f"esperados {expected}"
        )


def test_mf_breakpoints_match_design_doc():
    """Contrato con el diseño del subsistema de severidad.

    Si se recalibran los cortes, este test debe
    actualizarse junto con el código: la falla aquí señaliza que el
    documento de diseño y la implementación deben converger.
    """
    exp = build_exploitability_antecedent()
    imp = build_impact_antecedent()
    sev = build_severity_consequent()

    cases = [
        (exp, "low", [0.1, 0.1, 1.4]),
        (exp, "medium", [0.9, 2.0, 3.2]),
        (exp, "high", [2.5, 3.9, 3.9]),
        (imp, "low", [0.0, 0.0, 2.5]),
        (imp, "medium", [1.5, 3.5, 5.0]),
        (imp, "high", [4.0, 6.0, 6.0]),
        (sev, "low", [0.0, 0.0, 4.0]),
        (sev, "medium", [3.0, 5.0, 7.0]),
        (sev, "high", [6.0, 10.0, 10.0]),
    ]

    for var, label, breakpoints in cases:
        expected_mf = fuzz.trimf(var.universe, breakpoints)
        actual_mf = var.terms[label].mf
        assert np.allclose(actual_mf, expected_mf), (
            f"{var.label}[{label}] no coincide con breakpoints {breakpoints}"
        )


def test_mf_coverage_no_gaps():
    """Cada punto del universo está cubierto por alguna MF con μ > 0.

    Sin esto, los inputs que cayeran en un hueco producirían NaN en la
    defuzzificación porque ninguna regla se activaría con ellos.
    """
    for var in (
        build_exploitability_antecedent(),
        build_impact_antecedent(),
        build_severity_consequent(),
    ):
        lo, hi = float(var.universe[0]), float(var.universe[-1])
        points = np.linspace(lo, hi, 1000)
        memberships = np.stack(
            [
                fuzz.interp_membership(var.universe, var.terms[label].mf, points)
                for label in ("low", "medium", "high")
            ]
        )
        max_membership = memberships.max(axis=0)
        gap_mask = max_membership <= 0
        assert not gap_mask.any(), (
            f"{var.label}: hay puntos sin cobertura: "
            f"primer hueco en x={points[gap_mask][0]:.3f}"
        )


# --- -----------------------------------------------------------


def test_build_severity_fis_constructs():
    fis = build_severity_fis()
    assert fis.antecedent_names == ["exploitability_subscore", "impact_subscore"]
    assert fis.consequent_name == "severity"


def test_severity_at_log4shell_is_high():
    """CVE-2021-44228: exp=3.9, imp=6.0 (máximos). Esperado: severity alto.

    Valores documentados en el diseño del dataset.
    """
    fis = build_severity_fis()
    sev = fis.infer(exploitability_subscore=3.9, impact_subscore=6.0)
    assert math.isfinite(sev)
    assert sev >= 7.0, f"esperado severity >= 7.0 para Log4Shell, obtenido {sev:.3f}"


def test_severity_at_minimum_is_low():
    """Extremos mínimos del dominio: exp=0.1, imp=0.0. Esperado: severity bajo."""
    fis = build_severity_fis()
    sev = fis.infer(exploitability_subscore=0.1, impact_subscore=0.0)
    assert math.isfinite(sev)
    assert sev <= 3.0, f"esperado severity <= 3.0 para mínimos, obtenido {sev:.3f}"


def test_severity_at_cross_low_high_is_medium():
    """Caso discriminante de B vs A: exp=0.1 (low), imp=6.0 (high) → medium.

    Con la matriz B (elegida) las cruzadas suben a medium, así que la severity
    debe caer en el rango medio del consecuente. Con la matriz A (min puro)
    este mismo input daría severity en torno a 2: si alguien revierte a A,
    este test falla.
    """
    fis = build_severity_fis()
    sev = fis.infer(exploitability_subscore=0.1, impact_subscore=6.0)
    assert math.isfinite(sev)
    assert 3.5 < sev < 6.5, (
        f"esperado severity en zona medium (3.5, 6.5) por matriz B, "
        f"obtenido {sev:.3f}: ¿reglas degradadas a min puro (A)?"
    )


def test_compute_severity_adds_column_without_mutation():
    """`compute_severity` añade `severity` y no muta el input."""
    df_in = pd.DataFrame(
        {
            "exploitability_subscore": [0.1, 2.0, 3.9],
            "impact_subscore": [0.0, 3.5, 6.0],
        }
    )
    df_out = compute_severity(df_in)

    # No mutación del input.
    assert "severity" not in df_in.columns

    # Salida con columna, 3 floats finitos.
    assert "severity" in df_out.columns
    assert len(df_out) == 3
    assert df_out["severity"].apply(math.isfinite).all()

    # Monotonía ordinal: inputs crecientes en ambos ejes → severity creciente.
    sev_values = df_out["severity"].tolist()
    assert sev_values[0] < sev_values[1] < sev_values[2], (
        f"esperada monotonía estricta, obtenido {sev_values}"
    )
