"""Tests para `nvd_to_dataframe`.

No mockean red: la función solo lee disco. Se usa `tmp_path` para
simular `data/raw/nvd/` con páginas JSON sintéticas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import nvd


def _cve_with_v31(cve_id: str, published: str = "2022-06-01T00:00:00.000") -> dict:
    """CVE con CVSS v3.1 (subscores y configurations incluidos)."""
    return {
        "id": cve_id,
        "published": published,
        "lastModified": "2024-01-01T00:00:00.000",
        "descriptions": [{"lang": "en", "value": "Test v3.1 CVE"}],
        "metrics": {
            "cvssMetricV31": [
                {
                    "cvssData": {
                        "version": "3.1",
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                        "baseScore": 9.8,
                    },
                    "exploitabilityScore": 3.9,
                    "impactScore": 5.9,
                }
            ]
        },
        "configurations": [{"nodes": []}, {"nodes": []}],
    }


def _cve_with_v30(cve_id: str, published: str = "2022-06-01T00:00:00.000") -> dict:
    """CVE con CVSS v3.0 (sin v3.1)."""
    return {
        "id": cve_id,
        "published": published,
        "lastModified": "2024-01-01T00:00:00.000",
        "descriptions": [{"lang": "en", "value": "Test v3.0 CVE"}],
        "metrics": {
            "cvssMetricV30": [
                {
                    "cvssData": {
                        "version": "3.0",
                        "vectorString": "CVSS:3.0/AV:L/AC:H/PR:L/UI:R/S:U/C:L/I:L/A:N",
                        "baseScore": 4.2,
                    },
                    "exploitabilityScore": 0.8,
                    "impactScore": 3.4,
                }
            ]
        },
        "configurations": [{"nodes": []}],
    }


def _cve_with_v2_only(cve_id: str, published: str = "2022-06-01T00:00:00.000") -> dict:
    """CVE sin CVSS v3 (caso solo-v2 / RESERVED / REJECTED)."""
    return {
        "id": cve_id,
        "published": published,
        "lastModified": "2024-01-01T00:00:00.000",
        "descriptions": [{"lang": "en", "value": "Legacy v2 only"}],
        "metrics": {
            "cvssMetricV2": [
                {"cvssData": {"version": "2.0", "baseScore": 5.0}}
            ]
        },
        "configurations": [],
    }


def _write_fake_page(raw_dir: Path, cves: list[dict], page_idx: int = 0) -> Path:
    """Escribe una página JSON con la estructura `vulnerabilities[].cve.*`."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"cves_2022-01-01_2022-12-31_page{page_idx}.json"
    payload = {
        "resultsPerPage": 2000,
        "startIndex": 0,
        "totalResults": len(cves),
        "vulnerabilities": [{"cve": cve} for cve in cves],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_filters_only_v3(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(
        raw_dir,
        [
            _cve_with_v31("CVE-2022-0001"),
            _cve_with_v30("CVE-2022-0002"),
            _cve_with_v2_only("CVE-2022-0003"),
        ],
    )

    df, stats = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=None)

    assert stats["total_loaded"] == 3
    assert stats["dropped_no_v3"] == 1
    assert stats["dropped_out_of_range"] == 0
    assert stats["kept"] == 2
    assert len(df) == 2
    assert set(df["cve_id"]) == {"CVE-2022-0001", "CVE-2022-0002"}


def test_year_range_filter(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(
        raw_dir,
        [
            _cve_with_v31("CVE-2021-9999", published="2021-12-01T00:00:00.000"),
            _cve_with_v31("CVE-2022-0001", published="2022-06-01T00:00:00.000"),
            _cve_with_v31("CVE-2024-0001", published="2024-01-15T00:00:00.000"),
        ],
    )

    df, stats = nvd.nvd_to_dataframe(
        raw_dir=raw_dir, year_range=(2022, 2023), output_path=None
    )

    assert stats["dropped_out_of_range"] == 2
    assert stats["kept"] == 1
    assert df["cve_id"].tolist() == ["CVE-2022-0001"]


def test_year_range_none_keeps_all_years(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(
        raw_dir,
        [
            _cve_with_v31("CVE-2021-9999", published="2021-12-01T00:00:00.000"),
            _cve_with_v31("CVE-2024-0001", published="2024-01-15T00:00:00.000"),
        ],
    )

    df, stats = nvd.nvd_to_dataframe(
        raw_dir=raw_dir, year_range=None, output_path=None
    )

    assert stats["dropped_out_of_range"] == 0
    assert stats["kept"] == 2
    assert set(df["cve_id"]) == {"CVE-2021-9999", "CVE-2024-0001"}


def test_schema_types(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(raw_dir, [_cve_with_v31("CVE-2022-0001")])

    df, _ = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=None)

    assert list(df.columns) == list(nvd.NVD_DATAFRAME_COLUMNS)
    assert pd.api.types.is_datetime64_any_dtype(df["published"])
    assert pd.api.types.is_float_dtype(df["cvss_v3_score"])
    assert pd.api.types.is_float_dtype(df["exploitability_subscore"])
    assert pd.api.types.is_float_dtype(df["impact_subscore"])
    assert pd.api.types.is_integer_dtype(df["n_cpe_total"])


def test_extracts_subscores_and_n_cpe(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(raw_dir, [_cve_with_v31("CVE-2022-0001")])

    df, _ = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=None)
    row = df.iloc[0]

    assert row["exploitability_subscore"] == pytest.approx(3.9)
    assert row["impact_subscore"] == pytest.approx(5.9)
    assert row["n_cpe_total"] == 2


def test_writes_parquet_and_roundtrips(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(
        raw_dir,
        [_cve_with_v31("CVE-2022-0001"), _cve_with_v30("CVE-2022-0002")],
    )
    out_path = tmp_path / "out" / "nvd.parquet"

    df, _ = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=out_path)

    assert out_path.exists()
    df_loaded = pd.read_parquet(out_path)
    assert len(df_loaded) == len(df)
    assert set(df_loaded["cve_id"]) == set(df["cve_id"])
    assert list(df_loaded.columns) == list(nvd.NVD_DATAFRAME_COLUMNS)


def test_empty_raw_dir_returns_empty(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    df, stats = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=None)

    assert stats == {
        "total_loaded": 0,
        "dropped_out_of_range": 0,
        "dropped_no_v3": 0,
        "kept": 0,
    }
    assert len(df) == 0
    assert list(df.columns) == list(nvd.NVD_DATAFRAME_COLUMNS)


def test_stats_dict_sums_correctly(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_fake_page(
        raw_dir,
        [
            _cve_with_v31("CVE-2022-0001"),
            _cve_with_v2_only("CVE-2022-0002"),
            _cve_with_v31("CVE-2024-0003", published="2024-06-01T00:00:00.000"),
        ],
    )

    _, stats = nvd.nvd_to_dataframe(
        raw_dir=raw_dir, year_range=(2022, 2023), output_path=None
    )

    assert stats["total_loaded"] == 3
    assert (
        stats["dropped_out_of_range"] + stats["dropped_no_v3"] + stats["kept"]
        == stats["total_loaded"]
    )


def test_ignores_files_outside_pattern(tmp_path: Path) -> None:
    """Glob `cves_*.json` no debe recoger `exploration/` ni otros JSON."""
    raw_dir = tmp_path / "raw"
    _write_fake_page(raw_dir, [_cve_with_v31("CVE-2022-0001")])
    (raw_dir / "exploration").mkdir()
    (raw_dir / "exploration" / "cves_2022-01-01_2022-12-31_page0.json").write_text(
        json.dumps({"vulnerabilities": [{"cve": _cve_with_v31("CVE-2022-9999")}]}),
        encoding="utf-8",
    )
    (raw_dir / "summary.json").write_text("{}", encoding="utf-8")

    df, _ = nvd.nvd_to_dataframe(raw_dir=raw_dir, output_path=None)

    assert df["cve_id"].tolist() == ["CVE-2022-0001"]
