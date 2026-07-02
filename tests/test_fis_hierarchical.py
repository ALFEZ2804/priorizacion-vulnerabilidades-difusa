"""Smoke tests del wrapper jerárquico.

Verifica el encadenado de los 3 sub-FIS + post-procesado KEV (`+2 con clip
[0, 10]`). Las reglas internas de cada sub-FIS están cubiertas por
`test_fis_severity.py`, `test_fis_threat.py` y `test_fis_priority.py`; aquí
nos centramos en la composición, el boost KEV y la propagación de NaN.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.fis.hierarchical import KEV_BOOST, PRIORITY_MAX, compute_priority


# Fila Log4Shell-like: subscores máximos, EPSS muy alto, log_cpe en la rampa
# alta de `wide` (log4j afecta a cientos de productos → muchos CPE), en KEV.
# La usan varios tests como caso canónico de "todo al máximo".
_LOG4SHELL = {
    "exploitability_subscore": 3.9,
    "impact_subscore": 6.0,
    "epss_percentile": 0.999,
    "epss_delta": 0.0,
    "log_cpe": 7.0,
    "in_kev": True,
}


def _frame(*rows: dict) -> pd.DataFrame:
    """Construye un DataFrame de las filas dadas, en el orden esperado."""
    return pd.DataFrame(list(rows))


def test_compute_priority_adds_four_columns_without_mutation():
    df_in = _frame(_LOG4SHELL)
    cols_before = list(df_in.columns)
    df_out = compute_priority(df_in)

    # Las cuatro columnas nuevas presentes en la salida y ausentes en el input.
    for col in ("severity", "threat", "priority_raw", "priority"):
        assert col not in df_in.columns, f"compute_priority mutó el input añadiendo {col}"
        assert col in df_out.columns, f"compute_priority no añadió {col}"
    assert list(df_in.columns) == cols_before


def test_compute_priority_in_unit_range():
    """Sobre una rejilla razonable de inputs, `priority` ∈ [0, 10]."""
    df = _frame(
        {  # mínimo
            "exploitability_subscore": 0.1, "impact_subscore": 0.0,
            "epss_percentile": 0.0, "epss_delta": 0.0,
            "log_cpe": 0.69, "in_kev": False,
        },
        {  # intermedio
            "exploitability_subscore": 2.0, "impact_subscore": 3.5,
            "epss_percentile": 0.5, "epss_delta": 0.0,
            "log_cpe": 2.5, "in_kev": False,
        },
        _LOG4SHELL,
    )
    out = compute_priority(df)
    finite = out["priority"].dropna()
    assert (finite >= 0.0).all() and (finite <= 10.0).all(), out["priority"].tolist()


def test_log4shell_with_kev_is_max_priority():
    """Subscores máximos + EPSS muy alto + KEV → priority cerca de 10."""
    out = compute_priority(_frame(_LOG4SHELL))
    priority = out["priority"].iloc[0]
    assert math.isfinite(priority)
    assert priority >= 9.5, f"esperado priority ≥ 9.5 para Log4Shell+KEV, obtenido {priority:.3f}"


def test_kev_boost_applied_when_in_kev():
    """Dos filas idénticas salvo `in_kev`: la KEV recibe exactamente el boost."""
    base = dict(_LOG4SHELL)
    base["epss_percentile"] = 0.5  # bajamos para que priority_raw no sature
    base["epss_delta"] = 0.0
    base["impact_subscore"] = 3.0
    base["exploitability_subscore"] = 2.0
    base["log_cpe"] = 2.0

    no_kev = dict(base, in_kev=False)
    yes_kev = dict(base, in_kev=True)
    out = compute_priority(_frame(no_kev, yes_kev))

    raw_no, raw_yes = out["priority_raw"].iloc[0], out["priority_raw"].iloc[1]
    pri_no, pri_yes = out["priority"].iloc[0], out["priority"].iloc[1]

    # priority_raw debe ser idéntico (mismas entradas del FIS-3).
    assert math.isclose(raw_no, raw_yes, abs_tol=1e-9)
    # La fila sin KEV no recibe boost.
    assert math.isclose(pri_no, raw_no, abs_tol=1e-9)
    # La fila con KEV recibe exactamente +KEV_BOOST (sin saturar).
    assert math.isclose(pri_yes, raw_yes + KEV_BOOST, abs_tol=1e-9), (
        f"esperado priority_yes ≈ {raw_yes + KEV_BOOST:.3f}, obtenido {pri_yes:.3f}"
    )


def test_kev_boost_clipped_at_ten():
    """Cuando priority_raw + boost > 10, `priority` se satura exactamente en 10."""
    out = compute_priority(_frame(_LOG4SHELL))
    raw = out["priority_raw"].iloc[0]
    pri = out["priority"].iloc[0]
    # Sanity: el caso está diseñado para forzar el clip (raw alto + boost).
    assert raw + KEV_BOOST > PRIORITY_MAX, (
        f"este test exige raw + KEV_BOOST > 10 para verificar el clip; raw={raw:.3f}"
    )
    assert math.isclose(pri, PRIORITY_MAX, abs_tol=1e-9), f"priority no clipado: {pri}"


def test_compute_priority_propagates_nan_when_epss_missing():
    """Fila con EPSS NaN → threat NaN → priority_raw NaN → priority NaN."""
    df = _frame(
        _LOG4SHELL,
        {  # epss_percentile NaN
            "exploitability_subscore": 3.0, "impact_subscore": 4.0,
            "epss_percentile": np.nan, "epss_delta": 0.0,
            "log_cpe": 2.0, "in_kev": False,
        },
        {  # epss_delta NaN
            "exploitability_subscore": 3.0, "impact_subscore": 4.0,
            "epss_percentile": 0.5, "epss_delta": np.nan,
            "log_cpe": 2.0, "in_kev": True,  # el boost no debe rescatar el NaN
        },
    )
    out = compute_priority(df)
    assert math.isfinite(out["priority"].iloc[0])
    assert math.isnan(out["priority"].iloc[1])
    assert math.isnan(out["priority"].iloc[2]), (
        "el boost KEV no debe convertir NaN en un número; la incertidumbre del FIS "
        "se preserva sobre la observación KEV"
    )


def test_kev_boost_preserves_order_within_kev():
    """Argumento arquitectónico: dos CVEs ambos en KEV con priority_raw distinto
    mantienen su orden relativo en priority (sin saturar)."""
    df = _frame(
        {  # priority_raw moderado
            "exploitability_subscore": 1.5, "impact_subscore": 2.0,
            "epss_percentile": 0.2, "epss_delta": 0.0,
            "log_cpe": 1.0, "in_kev": True,
        },
        {  # priority_raw más alto
            "exploitability_subscore": 3.0, "impact_subscore": 4.0,
            "epss_percentile": 0.6, "epss_delta": 0.0,
            "log_cpe": 2.5, "in_kev": True,
        },
    )
    out = compute_priority(df)
    raw_lo, raw_hi = out["priority_raw"].iloc[0], out["priority_raw"].iloc[1]
    pri_lo, pri_hi = out["priority"].iloc[0], out["priority"].iloc[1]
    assert raw_lo < raw_hi, f"sanity check fallido: raw {raw_lo:.3f} vs {raw_hi:.3f}"
    assert pri_lo < pri_hi, (
        f"el boost KEV uniforme debería preservar el orden, obtenido {pri_lo:.3f} vs {pri_hi:.3f}"
    )
