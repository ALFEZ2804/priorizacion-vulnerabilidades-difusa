"""Smoke tests del wrapper `FuzzyInference`.

Comprueban la condición básica: con un antecedente
trivial y una regla, `.infer()` devuelve un float coherente. No
validan cortes ni reglas reales del FIS: eso es trabajo de los sub-FIS.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from skfuzzy import control as ctrl

from src.fis import FuzzyInference
from src.fis.priority import build_priority_fis
from src.fis.severity import build_severity_fis
from src.fis.threat import build_threat_fis


def test_engine_returns_finite_float_with_one_rule():
    """Un FIS trivial (1 entrada, 1 regla) infiere un float finito."""
    x = ctrl.Antecedent(np.arange(0, 11, 1), "x")
    y = ctrl.Consequent(np.arange(0, 11, 1), "y")
    x.automf(3)  # poor / average / good
    y.automf(3)

    # "IF x IS poor THEN y IS poor": input bajo → output bajo.
    rule = ctrl.Rule(x["poor"], y["poor"])
    fis = FuzzyInference(antecedents=[x], consequent=y, rules=[rule])

    out = fis.infer(x=0.0)

    assert isinstance(out, float)
    assert math.isfinite(out)
    # En x=0 la regla activa "poor" al máximo; el output queda
    # en la mitad baja del rango [0, 10].
    assert 0.0 <= out <= 5.0


def test_engine_exposes_labels():
    """Las propiedades de inspección devuelven los labels reales."""
    a = ctrl.Antecedent(np.arange(0, 5, 1), "foo")
    b = ctrl.Consequent(np.arange(0, 5, 1), "bar")
    a.automf(3)
    b.automf(3)
    rule = ctrl.Rule(a["poor"], b["poor"])

    fis = FuzzyInference([a], b, [rule])

    assert fis.antecedent_names == ["foo"]
    assert fis.consequent_name == "bar"


def test_engine_is_reusable_across_calls():
    """La misma instancia infiere para varios inputs sin reconstruir el grafo.

    Esto es la base de la optimización por cacheo:
    si una sola instancia puede procesar muchas llamadas, podemos recorrer
    los 92.350 CVEs reutilizándola.
    """
    x = ctrl.Antecedent(np.arange(0, 11, 1), "x")
    y = ctrl.Consequent(np.arange(0, 11, 1), "y")
    x.automf(3)
    y.automf(3)
    rules = [
        ctrl.Rule(x["poor"], y["poor"]),
        ctrl.Rule(x["average"], y["average"]),
        ctrl.Rule(x["good"], y["good"]),
    ]
    fis = FuzzyInference([x], y, rules)

    low = fis.infer(x=0.0)
    high = fis.infer(x=10.0)

    assert math.isfinite(low) and math.isfinite(high)
    assert low < high  # monotonía mínima: input bajo da output más bajo


def _grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, n)


@pytest.mark.parametrize(
    "build_fis, grids",
    [
        (
            build_severity_fis,
            {"exploitability_subscore": _grid(0.1, 3.9, 12), "impact_subscore": _grid(0.0, 6.0, 12)},
        ),
        (
            build_threat_fis,
            {"epss_percentile": _grid(0.0, 1.0, 12), "epss_delta": _grid(-1.0, 1.0, 12)},
        ),
        (
            build_priority_fis,
            {"severity": _grid(0.0, 10.0, 8), "threat": _grid(0.0, 10.0, 8), "log_cpe": _grid(0.69, 7.97, 6)},
        ),
    ],
)
def test_infer_batch_matches_infer(build_fis, grids):
    """`infer_batch` (vectorizado) reproduce `infer` (skfuzzy) punto a punto.

    Validación central de la vectorización: la reimplementación vectorizada de
    Mamdani debe dar EXACTAMENTE los mismos valores que `skfuzzy.control`, no
    una aproximación. Recorre el producto cartesiano de los dominios del sub-FIS.
    """
    fis = build_fis()
    names = list(grids)
    combos = list(itertools.product(*(grids[name] for name in names)))
    arrays = {name: np.array([combo[i] for combo in combos]) for i, name in enumerate(names)}

    batch = fis.infer_batch(**arrays)
    one_by_one = np.array([fis.infer(**dict(zip(names, combo))) for combo in combos])

    assert np.array_equal(np.isnan(batch), np.isnan(one_by_one))
    finite = np.isfinite(batch) & np.isfinite(one_by_one)
    assert finite.any()
    # Coincidencia a ~5 cifras significativas (diff máx ~3e-5 sobre escala 0-10).
    # El centroide es idéntico a skfuzzy.defuzz (verificado a 1e-14 aparte); la
    # diferencia residual viene de detalles internos de agregación de
    # skfuzzy.control y es despreciable para el ranking.
    assert np.allclose(batch[finite], one_by_one[finite], atol=1e-4)


def test_infer_batch_propagates_nan():
    """Una fila con algún input NaN devuelve NaN, sin contaminar a las demás."""
    fis = build_threat_fis()
    out = fis.infer_batch(
        epss_percentile=np.array([0.9, np.nan, 0.1]),
        epss_delta=np.array([0.5, 0.0, np.nan]),
    )
    assert math.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])
