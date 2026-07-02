"""Tests del pipeline end-to-end.

Cubre los dos pasos del pipeline:
  1. `fetch_cve("CVE-2021-44228")` con `requests.get` mockeado, validando
     el esquema plano de salida y los valores CVSS extremos del Log4Shell.
  2. Smoke contra el `unified.parquet` real de `data/processed/` para
     validar reproducibilidad. Si el parquet no existe (clone limpio o CI),
     el test se salta automáticamente.

El test de no-nulls en columnas críticas con fixtures sintéticos ya vive
en `tests/test_unify.py::test_no_nulls_in_critical_columns`: aquí lo
extendemos contra datos reales del pipeline.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import requests

from src.pipeline import nvd, unify


NVD_V31_RESPONSE_LOG4SHELL: dict = {
    "resultsPerPage": 1,
    "startIndex": 0,
    "totalResults": 1,
    "format": "NVD_CVE",
    "version": "2.0",
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2021-44228",
                "sourceIdentifier": "security@apache.org",
                "published": "2021-12-10T10:15:09.143",
                "lastModified": "2025-01-01T00:00:00.000",
                "vulnStatus": "Modified",
                "descriptions": [
                    {
                        "lang": "en",
                        "value": "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI features...",
                    },
                    {"lang": "es", "value": "Apache Log4j2 contiene..."},
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "source": "nvd@nist.gov",
                            "type": "Primary",
                            "cvssData": {
                                "version": "3.1",
                                "vectorString": (
                                    "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
                                ),
                                "baseScore": 10.0,
                                "baseSeverity": "CRITICAL",
                            },
                            "exploitabilityScore": 3.9,
                            "impactScore": 6.0,
                        }
                    ]
                },
                "configurations": [{"nodes": []}, {"nodes": []}, {"nodes": []}],
            }
        }
    ],
}


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[assignment]
            raise err

    def json(self) -> dict:
        return self._payload


def test_fetch_cve_log4shell_returns_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVD_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(
        nvd.requests,
        "get",
        lambda *a, **kw: _FakeResponse(200, NVD_V31_RESPONSE_LOG4SHELL),
    )

    result = nvd.fetch_cve("CVE-2021-44228")

    assert set(result.keys()) == set(nvd.NVD_DATAFRAME_COLUMNS)
    assert result["cve_id"] == "CVE-2021-44228"
    assert result["cvss_v3_score"] == 10.0
    assert result["exploitability_subscore"] == pytest.approx(3.9)
    assert result["impact_subscore"] == pytest.approx(6.0)
    assert result["n_cpe_total"] == 3


def test_fetch_cve_invalid_id_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid CVE ID format"):
        nvd.fetch_cve("not-a-cve")


@pytest.mark.skipif(
    not unify.PROCESSED_UNIFIED_PATH.exists(),
    reason=(
        "unified.parquet no presente en data/processed/: "
        "regenerar con scripts/build_dataset.py para activar este smoke"
    ),
)
def test_unified_parquet_smoke_schema_and_critical_nulls() -> None:
    df = pd.read_parquet(unify.PROCESSED_UNIFIED_PATH)

    assert list(df.columns) == list(unify.UNIFIED_DATAFRAME_COLUMNS)
    assert len(df) > 0

    critical_cols = (
        "cve_id",
        "published",
        "cvss_v3_score",
        "exploitability_subscore",
        "impact_subscore",
        "log_cpe",
        "in_kev",
        "target_year",
    )
    for col in critical_cols:
        assert df[col].notna().all(), f"Columna crítica con nulls: {col}"
