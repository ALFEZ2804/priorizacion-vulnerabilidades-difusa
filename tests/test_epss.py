"""Tests para el cliente EPSS.

Mockean `requests.get` (no pegar a Cyentia/FIRST) y `time.sleep` (no
esperar en backoff). El CSV de prueba se construye en memoria y se
gzippea para simular el formato real.
"""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.pipeline import epss


CSV_SAMPLE = (
    "#model_version:v2024.01.01,score_date:2023-12-31T00:00:00+0000\n"
    "cve,epss,percentile\n"
    "CVE-2021-44228,0.97450,0.99950\n"
    "CVE-2022-22965,0.94358,0.99800\n"
    "CVE-2022-0001,0.00100,0.10000\n"
)


def _gzip_bytes(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


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
    monkeypatch.setattr(epss.time, "sleep", lambda _s: None)


def test_parse_csv_casts_floats(tmp_path: Path) -> None:
    csv_path = tmp_path / "snap.csv.gz"
    csv_path.write_bytes(_gzip_bytes(CSV_SAMPLE))

    df = epss._parse_epss_csv(csv_path)

    assert df["epss"].dtype == float
    assert df["epss_percentile"].dtype == float
    assert df.loc[df["cve_id"] == "CVE-2021-44228", "epss"].iloc[0] == pytest.approx(0.9745)


def test_parse_csv_skips_header_comment(tmp_path: Path) -> None:
    csv_path = tmp_path / "snap.csv.gz"
    csv_path.write_bytes(_gzip_bytes(CSV_SAMPLE))

    df = epss._parse_epss_csv(csv_path)

    assert len(df) == 3
    assert set(df.columns) == {"cve_id", "epss", "epss_percentile"}
    assert "model_version" not in df["cve_id"].iloc[0]


def test_fetch_snapshot_writes_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(
        epss.requests,
        "get",
        lambda *a, **kw: _FakeResponse(200, _gzip_bytes(CSV_SAMPLE)),
    )

    df = epss.fetch_epss_snapshot(date(2023, 12, 31), raw_dir=raw_dir)

    csv_path = raw_dir / "epss_scores-2023-12-31.csv.gz"
    assert csv_path.exists()
    assert list(df.columns) == list(epss.EPSS_DATAFRAME_COLUMNS)
    assert (df["epss_date"] == "2023-12-31").all()
    assert len(df) == 3


def test_fetch_snapshot_skips_if_raw_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "epss_scores-2023-12-31.csv.gz"
    csv_path.write_bytes(_gzip_bytes(CSV_SAMPLE))

    called: list[str] = []

    def fail_if_called(*_a, **_kw):
        called.append("requests.get")
        raise AssertionError("debería haber reusado el CSV en disco")

    monkeypatch.setattr(epss.requests, "get", fail_if_called)

    df = epss.fetch_epss_snapshot(date(2023, 12, 31), raw_dir=raw_dir)

    assert called == []
    assert len(df) == 3


def test_backoff_on_5xx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            _FakeResponse(503),
            _FakeResponse(503),
            _FakeResponse(200, _gzip_bytes(CSV_SAMPLE)),
        ]
    )
    monkeypatch.setattr(epss.requests, "get", lambda *a, **kw: next(responses))

    raw_dir = tmp_path / "raw"
    df = epss.fetch_epss_snapshot(date(2023, 12, 31), raw_dir=raw_dir)

    assert len(df) == 3


def test_404_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(epss.requests, "get", lambda *a, **kw: _FakeResponse(404))

    with pytest.raises(requests.HTTPError):
        epss.fetch_epss_snapshot(date(1999, 1, 1), raw_dir=tmp_path / "raw")


def test_build_parquets_creates_one_per_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        epss.requests,
        "get",
        lambda *a, **kw: _FakeResponse(200, _gzip_bytes(CSV_SAMPLE)),
    )

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"

    paths = epss.build_epss_parquets(
        snapshot_dates=[date(2023, 12, 31), date(2024, 12, 31)],
        output_dir=out_dir,
        raw_dir=raw_dir,
    )

    assert set(paths.keys()) == {date(2023, 12, 31), date(2024, 12, 31)}
    for path in paths.values():
        assert path.exists()
        df = pd.read_parquet(path)
        assert list(df.columns) == list(epss.EPSS_DATAFRAME_COLUMNS)
        assert len(df) == 3
