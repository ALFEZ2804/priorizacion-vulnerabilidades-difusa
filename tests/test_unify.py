"""Tests para el unificador NVD + EPSS (2 snapshots) + KEV.

Construyen los DataFrames de entrada en memoria (sin red, sin disco salvo
el `tmp_path` del roundtrip) y verifican joins, columnas derivadas
(`epss_delta`, `log_cpe`, `in_kev`, `target_year`) y ausencia de leakage
temporal en `in_kev`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import unify


def _make_nvd_df() -> pd.DataFrame:
    """3 CVEs con CVSS v3 válido y subscores, publicados entre 2022 y 2023."""
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-0001", "CVE-2023-0002", "CVE-2023-0003"],
            "published": pd.to_datetime(
                ["2022-01-15", "2023-03-10", "2023-07-20"]
            ),
            "cvss_v3_score": [7.5, 9.8, 5.3],
            "exploitability_subscore": [2.8, 3.9, 1.6],
            "impact_subscore": [3.6, 5.9, 3.4],
            "n_cpe_total": [0, 10, 4],
        }
    )


def _make_epss_2023_df() -> pd.DataFrame:
    """EPSS 2023-12-31 solo para 2 de los 3 CVEs (cobertura parcial)."""
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-0001", "CVE-2023-0002"],
            "epss": [0.05, 0.92],
            "epss_percentile": [0.50, 0.99],
            "epss_date": ["2023-12-31", "2023-12-31"],
        }
    )


def _make_epss_2024_df() -> pd.DataFrame:
    """EPSS 2024-12-31. Combinado con 2023 fija los deltas testeables:

    - CVE-2022-0001: 0.20 − 0.05 = +0.15.
    - CVE-2023-0002: 0.50 − 0.92 = −0.42 (puede bajar EPSS).
    - CVE-2023-0003: solo en 2024 → delta = NaN (falta 2023).
    """
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-0001", "CVE-2023-0002", "CVE-2023-0003"],
            "epss": [0.20, 0.50, 0.30],
            "epss_percentile": [0.80, 0.85, 0.70],
            "epss_date": ["2024-12-31", "2024-12-31", "2024-12-31"],
        }
    )


def _make_kev_df() -> pd.DataFrame:
    """KEV con tres filas que cubren los tres casos de borde temporal:

    - CVE-2022-0001: date_added 2022-01-10 (anterior al freeze) → in_kev=True.
    - CVE-2023-0002: date_added 2023-06-01 (anterior al freeze) → in_kev=True.
    - CVE-2023-0003: date_added 2024-03-01 (posterior al freeze) → in_kev=False.
    """
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-0001", "CVE-2023-0002", "CVE-2023-0003"],
            "date_added": pd.to_datetime(
                ["2022-01-10", "2023-06-01", "2024-03-01"]
            ),
            "due_date": pd.to_datetime(
                ["2022-02-01", "2023-07-01", "2024-04-01"]
            ),
            "vendor_project": ["v1", "v2", "v3"],
            "product": ["p1", "p2", "p3"],
        }
    )


def _make_exploitdb_df() -> pd.DataFrame:
    """EDB con dos casos: un exploit pre-freeze y otro en ventana de test.

    - CVE-2022-0001: primer exploit 2021-05-01 (ya conocido a 2023-12-31).
    - CVE-2023-0003: primer exploit 2024-09-01 (nuevo en la ventana de test).
    - CVE-2023-0002: sin entrada en EDB → edb_first_published = NaT.
    """
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-0001", "CVE-2023-0003"],
            "edb_first_published": pd.to_datetime(["2021-05-01", "2024-09-01"]),
        }
    )


def _build(
    *, kev: pd.DataFrame | None = None, exploitdb: pd.DataFrame | None = "default"
) -> pd.DataFrame:
    edb = _make_exploitdb_df() if isinstance(exploitdb, str) else exploitdb
    return unify.build_unified_dataset(
        _make_nvd_df(),
        _make_epss_2023_df(),
        _make_epss_2024_df(),
        _make_kev_df() if kev is None else kev,
        edb,
    )


def test_columns_and_shape() -> None:
    df = _build()

    assert len(df) == 3
    assert list(df.columns) == list(unify.UNIFIED_DATAFRAME_COLUMNS)
    assert len(df.columns) == 12


def test_edb_join_populates_first_published() -> None:
    df = _build().set_index("cve_id")

    assert df.loc["CVE-2022-0001", "edb_first_published"] == pd.Timestamp("2021-05-01")
    assert df.loc["CVE-2023-0003", "edb_first_published"] == pd.Timestamp("2024-09-01")
    # sin entrada en EDB → NaT
    assert pd.isna(df.loc["CVE-2023-0002", "edb_first_published"])


def test_edb_absent_is_nat_column() -> None:
    df = _build(exploitdb=None)

    assert "edb_first_published" in df.columns
    assert df["edb_first_published"].isna().all()
    assert pd.api.types.is_datetime64_any_dtype(df["edb_first_published"])


def test_epss_join_partial_coverage() -> None:
    df = _build()

    assert df["epss_percentile"].notna().sum() == 2
    assert df.loc[df["cve_id"] == "CVE-2023-0003", "epss_percentile"].isna().all()


def test_epss_delta_calculation() -> None:
    df = _build()

    by_cve = df.set_index("cve_id")
    assert by_cve.loc["CVE-2022-0001", "epss_delta"] == pytest.approx(0.15)
    assert by_cve.loc["CVE-2023-0002", "epss_delta"] == pytest.approx(-0.42)
    # CVE-2023-0003 solo está en 2024 → delta no calculable
    assert pd.isna(by_cve.loc["CVE-2023-0003", "epss_delta"])


def test_log_cpe_uses_log1p() -> None:
    df = _build()

    by_cve = df.set_index("cve_id")
    assert by_cve.loc["CVE-2022-0001", "log_cpe"] == pytest.approx(np.log1p(0))
    assert by_cve.loc["CVE-2023-0002", "log_cpe"] == pytest.approx(np.log1p(10))
    assert by_cve.loc["CVE-2023-0003", "log_cpe"] == pytest.approx(np.log1p(4))


def test_in_kev_freeze_excludes_post_freeze_entries() -> None:
    df = _build()

    row_2024 = df[df["cve_id"] == "CVE-2023-0003"].iloc[0]
    assert bool(row_2024["in_kev"]) is False
    assert pd.notna(row_2024["kev_date_added"])
    assert row_2024["kev_date_added"] == pd.Timestamp("2024-03-01")


def test_in_kev_true_for_pre_freeze_entries() -> None:
    df = _build()

    by_cve = df.set_index("cve_id")
    assert bool(by_cve.loc["CVE-2022-0001", "in_kev"]) is True
    assert bool(by_cve.loc["CVE-2023-0002", "in_kev"]) is True


def test_in_kev_false_when_no_kev_match() -> None:
    df = _build(kev=_make_kev_df().iloc[:0])

    assert (df["in_kev"] == False).all()  # noqa: E712
    assert df["kev_date_added"].isna().all()


def test_target_year_extracted_from_published() -> None:
    df = _build()

    years = dict(zip(df["cve_id"], df["target_year"]))
    assert years["CVE-2022-0001"] == 2022
    assert years["CVE-2023-0002"] == 2023
    assert years["CVE-2023-0003"] == 2023


def test_no_nulls_in_critical_columns() -> None:
    df = _build()

    for col in (
        "cve_id",
        "published",
        "cvss_v3_score",
        "exploitability_subscore",
        "impact_subscore",
        "log_cpe",
        "target_year",
    ):
        assert df[col].notna().all(), f"nulls inesperados en {col}"


def test_roundtrip_parquet(tmp_path: Path) -> None:
    df = _build()
    out = tmp_path / "unified.parquet"

    unify.save_unified(df, out)
    loaded = pd.read_parquet(out)

    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(unify.UNIFIED_DATAFRAME_COLUMNS)
    assert loaded["in_kev"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(loaded["kev_date_added"])


def test_save_dataset_csv_roundtrip(tmp_path: Path) -> None:
    df = _build()
    out = tmp_path / "dataset.csv"

    unify.save_dataset_csv(df, out)
    loaded = pd.read_csv(
        out, parse_dates=["published", "kev_date_added"]
    )

    assert list(loaded.columns) == list(unify.UNIFIED_DATAFRAME_COLUMNS)
    assert len(loaded) == len(df)
    # in_kev se serializa como True/False y pandas lo recupera como bool
    assert loaded["in_kev"].dtype == bool


def test_build_unified_orchestrator(tmp_path: Path) -> None:
    nvd_path = tmp_path / "nvd.parquet"
    epss_2023_path = tmp_path / "epss_2023.parquet"
    epss_2024_path = tmp_path / "epss_2024.parquet"
    kev_path = tmp_path / "kev.parquet"
    exploitdb_path = tmp_path / "exploitdb.parquet"
    out_path = tmp_path / "unified.parquet"
    csv_out_path = tmp_path / "dataset.csv"

    _make_nvd_df().to_parquet(nvd_path, compression="snappy")
    _make_epss_2023_df().to_parquet(epss_2023_path, compression="snappy")
    _make_epss_2024_df().to_parquet(epss_2024_path, compression="snappy")
    _make_kev_df().to_parquet(kev_path, compression="snappy")
    _make_exploitdb_df().to_parquet(exploitdb_path, compression="snappy")

    stats = unify.build_unified(
        nvd_path=nvd_path,
        epss_path=epss_2023_path,
        epss_2024_path=epss_2024_path,
        kev_path=kev_path,
        exploitdb_path=exploitdb_path,
        output_path=out_path,
        csv_output_path=csv_out_path,
    )

    assert out_path.exists()
    assert csv_out_path.exists()
    assert stats["n_rows"] == 3
    assert stats["in_kev_count"] == 2
    assert stats["epss_cov_pct"] == pytest.approx(66.67, abs=0.01)
    # 2 de 3 tienen ambos snapshots
    assert stats["epss_delta_cov_pct"] == pytest.approx(66.67, abs=0.01)


def test_load_sources_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        unify.load_sources(
            nvd_path=tmp_path / "no.parquet",
            epss_path=tmp_path / "no.parquet",
            epss_2024_path=tmp_path / "no.parquet",
            kev_path=tmp_path / "no.parquet",
        )
