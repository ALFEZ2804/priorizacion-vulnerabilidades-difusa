"""Tests para los splits temporales train/test.

Construyen un DataFrame `unified` de prueba en memoria con cuatro casos de
borde temporal y verifican: cardinalidad compartida train/test, derivación
de los labels (`y_train`, `y_test`), ausencia de leakage temporal en ambos
splits, roundtrip Parquet y un smoke contra `unified.parquet` real.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline import splits


def _make_unified_df() -> pd.DataFrame:
    """4 CVEs que cubren los casos de borde del split temporal.

    Con la ventana de test KEV [2024-01-01, 2025-12-31] (24 meses):

    - CVE-2022-A: KEV antes del freeze (2022-06-01) → positivo train,
      negativo test. EDB pre-freeze (2021-03-01): ya conocido, no nuevo.
    - CVE-2022-B: KEV en 2024 (2024-04-01) → negativo train, positivo test.
    - CVE-2023-C: sin entrada KEV pero primer exploit EDB en 2024-07-01 →
      negativo en y_test, **positivo solo en y_test_multi** (caso multi-source).
    - CVE-2023-D: KEV en 2025 (2025-02-01) → negativo train, positivo test
      (dentro de la ventana 2024-2025).
    """
    return pd.DataFrame(
        {
            "cve_id": ["CVE-2022-A", "CVE-2022-B", "CVE-2023-C", "CVE-2023-D"],
            "published": pd.to_datetime(
                ["2022-05-10", "2022-08-20", "2023-02-15", "2023-11-30"]
            ),
            "exploitability_subscore": [3.9, 2.8, 1.6, 3.9],
            "impact_subscore": [5.9, 4.7, 3.4, 5.9],
            "epss_percentile": [0.99, 0.80, 0.30, 0.95],
            "epss_delta": [0.10, -0.05, 0.01, 0.20],
            "log_cpe": [2.30, 1.79, 0.69, 3.04],
            "cvss_v3_score": [9.8, 7.5, 5.3, 8.1],
            "in_kev": [True, False, False, False],
            "kev_date_added": pd.to_datetime(
                ["2022-06-01", "2024-04-01", None, "2025-02-01"]
            ),
            "edb_first_published": pd.to_datetime(
                ["2021-03-01", None, "2024-07-01", None]
            ),
            "target_year": pd.array([2022, 2022, 2023, 2023], dtype="int16"),
        }
    )


def test_cardinalidad_y_orden() -> None:
    unified = _make_unified_df()
    train, test = splits.build_train_test_splits(unified)

    assert len(train) == len(unified) == 4
    assert len(test) == len(unified) == 4
    assert list(train["cve_id"]) == list(unified["cve_id"])
    assert list(test["cve_id"]) == list(unified["cve_id"])


def test_y_train_es_in_kev() -> None:
    unified = _make_unified_df()
    train, _ = splits.build_train_test_splits(unified)

    assert (train["y_train"] == unified["in_kev"]).all()
    assert int(train["y_train"].sum()) == 1
    assert train["y_train"].dtype == bool


def test_y_test_rango_2024_2025() -> None:
    unified = _make_unified_df()
    _, test = splits.build_train_test_splits(unified)

    positives = test.loc[test["y_test"], "cve_id"].tolist()
    assert positives == ["CVE-2022-B", "CVE-2023-D"]
    assert int(test["y_test"].sum()) == 2


def test_y_test_multi_incluye_edb() -> None:
    unified = _make_unified_df()
    _, test = splits.build_train_test_splits(unified)

    multi_pos = test.loc[test["y_test_multi"], "cve_id"].tolist()
    # B y D entran por KEV; C entra solo por EDB (nuevo en la ventana).
    assert multi_pos == ["CVE-2022-B", "CVE-2023-C", "CVE-2023-D"]
    assert int(test["y_test_multi"].sum()) == 3
    # CVE-2023-C es positivo multi pero no positivo KEV.
    c = test.set_index("cve_id").loc["CVE-2023-C"]
    assert bool(c["y_test_multi"]) is True
    assert bool(c["y_test"]) is False


def test_leakage_edb_fuera_de_ventana_lanza() -> None:
    # Marca un positivo solo-EDB con fecha fuera de la ventana → debe saltar.
    unified = _make_unified_df()
    train, test = splits.build_train_test_splits(unified)
    bad = test.copy()
    bad.loc[bad["cve_id"] == "CVE-2023-C", "edb_first_published"] = pd.Timestamp(
        "2019-01-01"
    )
    start, end = splits.KEV_TEST_RANGE
    with pytest.raises(ValueError, match="Leakage en test multi-source"):
        splits._assert_no_leakage(
            train, bad, splits.KEV_FREEZE_DATE, start, end
        )


def test_no_leakage_train() -> None:
    unified = _make_unified_df()
    train, _ = splits.build_train_test_splits(unified)

    pos_dates = train.loc[train["y_train"], "kev_date_added"]
    assert pos_dates.notna().all()
    assert pos_dates.max() <= splits.KEV_FREEZE_DATE


def test_no_leakage_test() -> None:
    unified = _make_unified_df()
    _, test = splits.build_train_test_splits(unified)

    pos_dates = test.loc[test["y_test"], "kev_date_added"]
    start, end = splits.KEV_TEST_RANGE
    assert pos_dates.min() >= start
    assert pos_dates.max() <= end


def test_features_identicas_entre_train_y_test() -> None:
    unified = _make_unified_df()
    train, test = splits.build_train_test_splits(unified)

    shared_cols = [c for c in unified.columns if c in train.columns and c in test.columns]
    pd.testing.assert_frame_equal(
        train[shared_cols].reset_index(drop=True),
        test[shared_cols].reset_index(drop=True),
    )


def test_missing_required_column_raises() -> None:
    unified = _make_unified_df().drop(columns=["in_kev"])
    with pytest.raises(ValueError, match="Columnas requeridas ausentes"):
        splits.build_train_test_splits(unified)


def test_roundtrip_parquet(tmp_path: Path) -> None:
    unified = _make_unified_df()
    train, test = splits.build_train_test_splits(unified)

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    splits.save_splits(train, test, train_path=train_path, test_path=test_path)

    loaded_train = pd.read_parquet(train_path)
    loaded_test = pd.read_parquet(test_path)

    assert loaded_train["y_train"].dtype == bool
    assert loaded_test["y_test"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(loaded_train["kev_date_added"])
    assert len(loaded_train) == len(loaded_test) == 4


def test_build_splits_orchestrator(tmp_path: Path) -> None:
    unified_path = tmp_path / "unified.parquet"
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    _make_unified_df().to_parquet(unified_path, compression="snappy")

    stats = splits.build_splits(
        unified_path=unified_path, train_path=train_path, test_path=test_path
    )

    assert train_path.exists() and test_path.exists()
    assert stats["n_rows"] == 4
    assert stats["n_train_positives"] == 1
    assert stats["n_test_positives"] == 2
    assert stats["n_test_multi_positives"] == 3
    assert stats["test_base_rate_pct"] == pytest.approx(50.0, abs=0.01)


@pytest.mark.skipif(
    not splits.PROCESSED_UNIFIED_PATH.exists(),
    reason=(
        "unified.parquet no presente en data/processed/: "
        "regenerar con scripts/build_dataset.py para activar este smoke"
    ),
)
def test_real_unified_split_smoke() -> None:
    unified = splits.load_unified()
    train, test = splits.build_train_test_splits(unified)

    assert len(train) == len(test) == len(unified)
    n_train_pos = int(train["y_train"].sum())
    n_test_pos = int(test["y_test"].sum())

    # Rangos con universo ampliado (2020-2023) y ventana
    # test 2024-2025: n_train=566, n_test=74. Tolerancia ±15% para absorber
    # actualizaciones futuras de KEV.
    assert 480 <= n_train_pos <= 650, f"in_kev fuera del rango esperado: {n_train_pos}"
    assert 50 <= n_test_pos <= 100, (
        f"positivos test fuera del rango esperado [50, 100]: {n_test_pos}"
    )

    leak_train = train.loc[train["y_train"], "kev_date_added"]
    assert leak_train.max() <= splits.KEV_FREEZE_DATE

    leak_test = test.loc[test["y_test"], "kev_date_added"]
    assert leak_test.min() >= splits.KEV_TEST_RANGE[0]
    assert leak_test.max() <= splits.KEV_TEST_RANGE[1]
