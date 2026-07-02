"""Tests para `fetch_cves_in_range`.

Mockean `requests.get` para no pegar a la API real y `time.sleep` para
no esperar entre páginas ni durante el backoff.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import requests

from src.pipeline import nvd


def _make_cve(cve_id: str = "CVE-2024-0001") -> dict:
    """CVE mínimo válido para `_parse_cve` (sin CVSS, sin configurations)."""
    return {
        "id": cve_id,
        "published": "2024-01-01T00:00:00.000",
        "lastModified": "2024-01-02T00:00:00.000",
        "descriptions": [{"lang": "en", "value": "Test CVE"}],
        "metrics": {},
        "configurations": [],
    }


def _make_payload(total_results: int, returned_count: int, start_index: int) -> dict:
    """Fabrica un response NVD con la forma exacta esperada."""
    return {
        "resultsPerPage": nvd.RESULTS_PER_PAGE,
        "startIndex": start_index,
        "totalResults": total_results,
        "format": "NVD_CVE",
        "version": "2.0",
        "timestamp": "2024-01-01T00:00:00.000",
        "vulnerabilities": [
            {"cve": _make_cve(f"CVE-2024-{i:04d}")} for i in range(returned_count)
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


@pytest.fixture(autouse=True)
def _api_key_and_no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configura `NVD_API_KEY` y anula `time.sleep` para todos los tests."""
    monkeypatch.setenv("NVD_API_KEY", "dummy-key")
    monkeypatch.setattr(nvd.time, "sleep", lambda _s: None)


@pytest.fixture
def isolated_raw_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirige `RAW_NVD_DIR` a un directorio temporal del test."""
    target = tmp_path / "raw_nvd"
    monkeypatch.setattr(nvd, "RAW_NVD_DIR", target)
    return target


def _make_get_mock(responses: list[_FakeResponse]) -> tuple[Any, list[dict]]:
    """Devuelve un mock de `requests.get` que entrega `responses` en orden y
    captura los params de cada llamada."""
    captured: list[dict] = []
    iterator = iter(responses)

    def fake_get(url: str, headers: dict, params: dict, timeout: int) -> _FakeResponse:
        captured.append(dict(params))
        return next(iterator)

    return fake_get, captured


def test_single_page(monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path) -> None:
    payload = _make_payload(total_results=10, returned_count=10, start_index=0)
    fake_get, captured = _make_get_mock([_FakeResponse(200, payload)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    cves = list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))

    assert len(cves) == 10
    assert len(captured) == 1
    assert captured[0]["startIndex"] == 0
    assert captured[0]["resultsPerPage"] == nvd.RESULTS_PER_PAGE
    assert captured[0]["pubStartDate"] == "2024-01-01T00:00:00.000"
    assert captured[0]["pubEndDate"] == "2024-01-31T23:59:59.999"


def test_multiple_pages(monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path) -> None:
    page0 = _make_payload(total_results=3500, returned_count=2000, start_index=0)
    page1 = _make_payload(total_results=3500, returned_count=1500, start_index=2000)
    fake_get, captured = _make_get_mock([_FakeResponse(200, page0), _FakeResponse(200, page1)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    cves = list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))

    assert len(cves) == 3500
    assert len(captured) == 2
    assert captured[0]["startIndex"] == 0
    assert captured[1]["startIndex"] == 2000


def test_total_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path
) -> None:
    bad = _make_payload(total_results=100, returned_count=50, start_index=0)
    fake_get, _ = _make_get_mock([_FakeResponse(200, bad)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="descargados 50 CVEs, esperados 100"):
        list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))


def test_backoff_on_429(monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path) -> None:
    payload = _make_payload(total_results=5, returned_count=5, start_index=0)
    fake_get, _ = _make_get_mock([_FakeResponse(429), _FakeResponse(200, payload)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    cves = list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))

    assert len(cves) == 5


def test_backoff_exhausted(monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path) -> None:
    responses = [_FakeResponse(429) for _ in range(nvd.MAX_RETRIES)]
    fake_get, _ = _make_get_mock(responses)
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError):
        list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))


def test_non_retryable_4xx_propagates(
    monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path
) -> None:
    """404 / 400 no deben reintentarse: son bug del cliente, no transitorios."""
    fake_get, _ = _make_get_mock([_FakeResponse(404)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError):
        list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))


def test_save_raw_writes_per_page(
    monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path
) -> None:
    page0 = _make_payload(total_results=3500, returned_count=2000, start_index=0)
    page1 = _make_payload(total_results=3500, returned_count=1500, start_index=2000)
    fake_get, _ = _make_get_mock([_FakeResponse(200, page0), _FakeResponse(200, page1)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=True))

    expected_files = {
        "cves_2024-01-01_2024-01-31_page0.json",
        "cves_2024-01-01_2024-01-31_page1.json",
    }
    actual_files = {p.name for p in isolated_raw_dir.iterdir()}
    assert actual_files == expected_files


def test_save_raw_disabled(
    monkeypatch: pytest.MonkeyPatch, isolated_raw_dir: Path
) -> None:
    payload = _make_payload(total_results=10, returned_count=10, start_index=0)
    fake_get, _ = _make_get_mock([_FakeResponse(200, payload)])
    monkeypatch.setattr(nvd.requests, "get", fake_get)

    list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))

    assert not isolated_raw_dir.exists() or not any(isolated_raw_dir.iterdir())


def test_invalid_range_raises(isolated_raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="debe ser <="):
        list(nvd.fetch_cves_in_range(date(2024, 2, 1), date(2024, 1, 1), save_raw=False))


def test_missing_api_key_warns(
    monkeypatch: pytest.MonkeyPatch,
    isolated_raw_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sin `NVD_API_KEY` no se aborta: se avisa y se llama a la API pública (sin header apiKey)."""
    monkeypatch.delenv("NVD_API_KEY", raising=False)

    captured_headers: list[dict] = []

    def fake_get(url: str, headers: dict, params: dict, timeout: int) -> _FakeResponse:
        captured_headers.append(dict(headers))
        return _FakeResponse(200, _make_payload(total_results=1, returned_count=1, start_index=0))

    monkeypatch.setattr(nvd.requests, "get", fake_get)

    with caplog.at_level(logging.WARNING):
        result = list(nvd.fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31), save_raw=False))

    assert len(result) == 1  # descarga sin clave
    assert captured_headers[0] == {}  # se llama sin el header apiKey
    assert any("NVD_API_KEY" in r.message for r in caplog.records)  # avisó de la falta de clave
