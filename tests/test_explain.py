"""Tests del módulo de explicación XAI (`src.fis.explain`).

Garantizan que la explicación es coherente con el motor de producción (la
prioridad explicada coincide con `compute_priority`) y que las tablas de reglas
del módulo no se desincronizan de las reglas reales de los sub-FIS.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.fis import compute_priority
from src.fis.explain import (
    PRIORITY_RULES,
    SEVERITY_RULES,
    THREAT_RULES,
    explain_cve,
    explain_severity,
)


def test_rule_tables_have_expected_sizes():
    assert len(SEVERITY_RULES) == 9
    assert len(THREAT_RULES) == 9
    assert len(PRIORITY_RULES) == 27


@pytest.mark.parametrize(
    "row",
    [
        dict(exploitability_subscore=3.9, impact_subscore=6.0, epss_percentile=0.99,
             epss_delta=0.0, log_cpe=7.0, in_kev=True),
        dict(exploitability_subscore=1.0, impact_subscore=2.0, epss_percentile=0.05,
             epss_delta=-0.5, log_cpe=0.69, in_kev=False),
        dict(exploitability_subscore=2.4, impact_subscore=4.6, epss_percentile=0.80,
             epss_delta=0.15, log_cpe=3.5, in_kev=False),
    ],
)
def test_explained_priority_matches_engine(row):
    """La prioridad de la explicación coincide con la del FIS de producción."""
    expected = compute_priority(pd.DataFrame([row]))["priority"].iloc[0]
    got = explain_cve(**row)["priority"]
    assert got == pytest.approx(float(expected), abs=0.05)


def test_fired_rules_sorted_and_bounded():
    e = explain_severity(3.9, 6.0)  # exploit alto, impact alto -> severidad alta
    assert e.dominant_label == "high"
    strengths = [r.strength for r in e.fired_rules]
    assert strengths == sorted(strengths, reverse=True)
    assert all(0.0 < s <= 1.0 for s in strengths)


def test_summary_is_human_readable():
    out = explain_cve(exploitability_subscore=3.9, impact_subscore=6.0,
                      epss_percentile=0.99, epss_delta=0.0, log_cpe=7.0, in_kev=True)
    assert "Prioridad" in out["summary"]
    assert "KEV" in out["summary"]  # el ajuste por explotación confirmada se menciona
    assert math.isfinite(out["priority"])
