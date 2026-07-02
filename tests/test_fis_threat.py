"""Smoke tests del FIS-2 Amenaza.

11.1: MFs (percentile trapezoidal, delta/threat triangulares), breakpoints
contra el diseño, cobertura sin huecos.
11.2: ensamblado del FIS, sanity de los criterios de la matriz B,
`compute_threat` sobre DataFrame (incluido el guard de NaN).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from src.fis.threat import (
    _PCT_A,
    _PCT_B,
    _PCT_C,
    _PCT_D,
    build_epss_delta_antecedent,
    build_epss_percentile_antecedent,
    build_threat_consequent,
    build_threat_fis,
    compute_threat,
)


def test_antecedents_have_expected_labels():
    pct = build_epss_percentile_antecedent()
    dlt = build_epss_delta_antecedent()
    assert pct.label == "epss_percentile"
    assert dlt.label == "epss_delta"


def test_consequent_has_expected_label():
    thr = build_threat_consequent()
    assert thr.label == "threat"


def test_terms_match_design():
    """Cada variable tiene exactamente las etiquetas del diseño."""
    assert set(build_epss_percentile_antecedent().terms.keys()) == {
        "low", "medium", "high",
    }
    assert set(build_epss_delta_antecedent().terms.keys()) == {
        "decreasing", "stable", "increasing",
    }
    assert set(build_threat_consequent().terms.keys()) == {"low", "medium", "high"}


def test_mf_breakpoints_match_design_doc():
    """Contrato con las MFs del FIS-2.

    `epss_percentile` tiene `low`/`medium` trapezoidales (4 vértices) con los
    cortes **calibrados** (`_PCT_A.._PCT_D`, de `scripts/optimize_fis.py`) y `high`
    **triangular** (no satura). `epss_delta` y `threat` son triangulares con sus
    cortes de dominio. Si se recalibran los cortes, basta con que cambien las
    constantes del módulo: este test las importa, no las duplica.
    """
    pct = build_epss_percentile_antecedent()
    dlt = build_epss_delta_antecedent()
    thr = build_threat_consequent()

    trap_cases = [
        (pct, "low", [0.0, 0.0, _PCT_A, _PCT_B]),
        (pct, "medium", [_PCT_A, _PCT_B, _PCT_C, _PCT_D]),
    ]
    for var, label, bps in trap_cases:
        expected = fuzz.trapmf(var.universe, bps)
        assert np.allclose(var.terms[label].mf, expected), (
            f"{var.label}[{label}] no coincide con trapezoide {bps}"
        )

    tri_cases = [
        (pct, "high", [_PCT_D, 1.0, 1.0]),
        (dlt, "decreasing", [-1.0, -1.0, -0.1]),
        (dlt, "stable", [-0.2, 0.0, 0.2]),
        (dlt, "increasing", [0.1, 1.0, 1.0]),
        (thr, "low", [0.0, 0.0, 4.0]),
        (thr, "medium", [3.0, 5.0, 7.0]),
        (thr, "high", [6.0, 10.0, 10.0]),
    ]
    for var, label, bps in tri_cases:
        expected = fuzz.trimf(var.universe, bps)
        assert np.allclose(var.terms[label].mf, expected), (
            f"{var.label}[{label}] no coincide con triángulo {bps}"
        )


def test_mf_coverage_no_gaps():
    """Cada punto del universo está cubierto por alguna MF con μ > 0."""
    variables = [
        (build_epss_percentile_antecedent(), ("low", "medium", "high")),
        (build_epss_delta_antecedent(), ("decreasing", "stable", "increasing")),
        (build_threat_consequent(), ("low", "medium", "high")),
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


def test_build_threat_fis_constructs():
    fis = build_threat_fis()
    assert fis.antecedent_names == ["epss_percentile", "epss_delta"]
    assert fis.consequent_name == "threat"


def test_high_percentile_dominates():
    """`percentile high` → threat high, aunque el delta caiga (regla high∧decreasing→high)."""
    fis = build_threat_fis()
    thr = fis.infer(epss_percentile=0.95, epss_delta=-0.5)
    assert math.isfinite(thr)
    assert thr >= 7.0, f"esperado threat >= 7.0 con percentile alto, obtenido {thr:.3f}"


def test_low_percentile_decreasing_is_low():
    """`percentile low` + `delta decreasing` → threat bajo."""
    fis = build_threat_fis()
    thr = fis.infer(epss_percentile=0.05, epss_delta=-0.5)
    assert math.isfinite(thr)
    assert thr <= 3.0, f"esperado threat <= 3.0, obtenido {thr:.3f}"


def test_delta_rescues_low_percentile():
    """Celda discriminante de B: percentile low + delta increasing → medium.

    Con la matriz B esta cruzada sube a `medium` (rescate por tendencia), así que
    threat debe caer en la zona media del consecuente, no en la baja.
    """
    fis = build_threat_fis()
    thr = fis.infer(epss_percentile=0.1, epss_delta=0.5)
    assert math.isfinite(thr)
    assert 3.5 < thr < 6.5, (
        f"esperado threat en zona medium (3.5, 6.5) por la celda 'delta rescata', "
        f"obtenido {thr:.3f}"
    )


def test_matches_design_doc_worked_example():
    """Reproduce el ejemplo numérico trabajado de la memoria.

    Con los cortes calibrados, percentile=0.80 cae en `medium` y delta=0.15
    (subiendo) activa la regla medium∧increasing→high; el resultado es
    threat ≈ 5.56. El código lo reproduce, lo que valida que la implementación
    es fiel al ejemplo trabajado a mano en la memoria.
    """
    fis = build_threat_fis()
    thr = fis.infer(epss_percentile=0.80, epss_delta=0.15)
    assert math.isfinite(thr)
    assert abs(thr - 5.56) < 0.2, (
        f"esperado threat ≈ 5.56 (ejemplo trabajado), obtenido {thr:.3f}"
    )


def test_compute_threat_adds_column_without_mutation():
    """`compute_threat` añade `threat`, no muta el input y respeta la monotonía."""
    # Percentiles en bandas distintas (low/medium/high con los cortes calibrados)
    # y delta neutro (stable) para aislar el efecto del percentile.
    df_in = pd.DataFrame(
        {
            "epss_percentile": [0.30, 0.80, 0.97],
            "epss_delta": [0.0, 0.0, 0.0],
        }
    )
    df_out = compute_threat(df_in)

    assert "threat" not in df_in.columns
    assert "threat" in df_out.columns
    assert len(df_out) == 3
    assert df_out["threat"].apply(math.isfinite).all()

    threats = df_out["threat"].tolist()
    assert threats[0] < threats[1] < threats[2], (
        f"esperada monotonía con percentile creciente, obtenido {threats}"
    )


def test_compute_threat_propagates_nan():
    """Filas sin EPSS reciben threat=NaN sin romper el motor."""
    df_in = pd.DataFrame(
        {
            "epss_percentile": [0.9, np.nan, 0.5],
            "epss_delta": [0.1, 0.1, np.nan],
        }
    )
    df_out = compute_threat(df_in)
    assert math.isfinite(df_out["threat"].iloc[0])
    assert math.isnan(df_out["threat"].iloc[1])
    assert math.isnan(df_out["threat"].iloc[2])
