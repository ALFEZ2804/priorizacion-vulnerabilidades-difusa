"""Tests para el cliente KEV.

Mockean `requests.get` (no pegar a CISA) y `time.sleep` (no esperar en
backoff). El JSON KEV de prueba se construye en memoria con tres
vulnerabilidades, una de ellas CVE-2021-44228 (Log4Shell) por coherencia
con el CSV de `test_epss.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.pipeline import kev


KEV_JSON_SAMPLE: dict = {
    "title": "CISA Catalog of Known Exploited Vulnerabilities",
    "catalogVersion": "2024.01.01",
    "dateReleased": "2024-01-01T00:00:00.000Z",
    "count": 3,
    "vulnerabilities": [
        {
            "cveID": "CVE-2021-44228",
            "vendorProject": "Apache",
            "product": "Log4j2",
            "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "dateAdded": "2021-12-10",
            "shortDescription": "Apache Log4j2 contains a vulnerability...",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2021-12-24",
            "knownRansomwareCampaignUse": "Known",
            "notes": "",
        },
        {
            "cveID": "CVE-2022-22965",
            "vendorProject": "VMware Tanzu",
            "product": "Spring Framework",
            "vulnerabilityName": "Spring Framework JDK 9+ RCE Vulnerability",
            "dateAdded": "2022-04-04",
            "shortDescription": "Spring Framework contains...",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2022-04-25",
            "knownRansomwareCampaignUse": "Known",
            "notes": "",
        },
        {
            "cveID": "CVE-2017-0144",
            "vendorProject": "Microsoft",
            "product": "SMBv1 Server",
            "vulnerabilityName": "Microsoft SMBv1 Remote Code Execution Vulnerability",
            "dateAdded": "2022-03-25",
            "shortDescription": "SMBv1 in Microsoft Windows...",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2022-04-15",
            "knownRansomwareCampaignUse": "Known",
            "notes": "",
        },
    ],
}


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[assignment]
            raise err


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kev.time, "sleep", lambda _s: None)


def test_parse_kev_json_renames_and_casts_dates(tmp_path: Path) -> None:
    json_path = tmp_path / "kev.json"
    json_path.write_bytes(_json_bytes(KEV_JSON_SAMPLE))

    df = kev._parse_kev_json(json_path)

    assert list(df.columns) == list(kev.KEV_DATAFRAME_COLUMNS)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["date_added"])
    assert pd.api.types.is_datetime64_any_dtype(df["due_date"])
    log4shell = df.loc[df["cve_id"] == "CVE-2021-44228"].iloc[0]
    assert log4shell["vendor_project"] == "Apache"
    assert log4shell["product"] == "Log4j2"
    assert log4shell["date_added"] == pd.Timestamp("2021-12-10")


def test_fetch_kev_writes_raw_and_returns_dataframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(
        kev.requests,
        "get",
        lambda *a, **kw: _FakeResponse(200, _json_bytes(KEV_JSON_SAMPLE)),
    )

    df = kev.fetch_kev(raw_dir=raw_dir)

    json_path = raw_dir / kev.RAW_KEV_FILENAME
    assert json_path.exists()
    assert list(df.columns) == list(kev.KEV_DATAFRAME_COLUMNS)
    assert len(df) == 3
    assert set(df["cve_id"]) == {"CVE-2021-44228", "CVE-2022-22965", "CVE-2017-0144"}


def test_fetch_kev_reuses_raw_if_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / kev.RAW_KEV_FILENAME).write_bytes(_json_bytes(KEV_JSON_SAMPLE))

    called: list[str] = []

    def fail_if_called(*_a, **_kw):
        called.append("requests.get")
        raise AssertionError("debería haber reusado el JSON en disco")

    monkeypatch.setattr(kev.requests, "get", fail_if_called)

    df = kev.fetch_kev(raw_dir=raw_dir)

    assert called == []
    assert len(df) == 3


def test_fetch_kev_backoff_on_5xx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter(
        [
            _FakeResponse(503),
            _FakeResponse(503),
            _FakeResponse(200, _json_bytes(KEV_JSON_SAMPLE)),
        ]
    )
    monkeypatch.setattr(kev.requests, "get", lambda *a, **kw: next(responses))

    df = kev.fetch_kev(raw_dir=tmp_path / "raw")

    assert len(df) == 3


def test_fetch_kev_404_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = {"n": 0}

    def get_404(*_a, **_kw):
        call_count["n"] += 1
        return _FakeResponse(404)

    monkeypatch.setattr(kev.requests, "get", get_404)

    with pytest.raises(requests.HTTPError):
        kev.fetch_kev(raw_dir=tmp_path / "raw")

    assert call_count["n"] == 1


def test_build_kev_parquet_writes_snappy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        kev.requests,
        "get",
        lambda *a, **kw: _FakeResponse(200, _json_bytes(KEV_JSON_SAMPLE)),
    )

    output_path = tmp_path / "out" / "kev.parquet"
    raw_dir = tmp_path / "raw"

    written = kev.build_kev_parquet(output_path=output_path, raw_dir=raw_dir)

    assert written == output_path
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert list(df.columns) == list(kev.KEV_DATAFRAME_COLUMNS)
    assert len(df) == 3
