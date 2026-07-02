"""Splits temporales train/test.

Construye dos Parquets a partir de `unified.parquet` que materializan el
setup experimental:

- `train.parquet`: los CVEs publicados en 2020-2023, con features
  congeladas a 2023-12-31 y etiqueta `y_train = in_kev` (estado de KEV al
  cierre del periodo de train).
- `test.parquet`: el mismo universo de CVEs, mismas features, etiqueta
  `y_test = kev_date_added ∈ [2024-01-01, 2025-12-31]`. Responde a la
  pregunta operativa "al cierre de 2023, ¿qué CVEs serán añadidos a KEV
  durante los 24 meses siguientes?". Incluye además `y_test_multi`, el
  ground truth secundario: `y_test OR (primer exploit en EDB
  publicado en la misma ventana)`. KEV sigue siendo el principal; el
  filtro del universo multi-source (excluir los ya conocidos por EDB antes
  del freeze) se aplica en `src/evaluation/dataset.py`.

No es un split por filas: train y test comparten población e inputs; lo
único que cambia es el target. Esta forma de evaluación retrospectiva es
la que valida los baselines (CVSS, EPSS, RF) y el FIS sin leakage temporal.

Universo y ventana ampliados respecto a la versión inicial
(2022-2023 × KEV-2024) para entrar en el tramo cómodo de poder estadístico
en la evaluación (~62-67 positivos test estimados frente a 27 iniciales).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROCESSED_UNIFIED_PATH = Path("data/processed/unified.parquet")
PROCESSED_TRAIN_PATH = Path("data/processed/train.parquet")
PROCESSED_TEST_PATH = Path("data/processed/test.parquet")

KEV_FREEZE_DATE = pd.Timestamp("2023-12-31")
KEV_TEST_RANGE: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2025-12-31"),
)

REQUIRED_UNIFIED_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "in_kev",
    "kev_date_added",
    "edb_first_published",
)

logger = logging.getLogger(__name__)


def load_unified(path: Path = PROCESSED_UNIFIED_PATH) -> pd.DataFrame:
    """Lee `unified.parquet` y devuelve el DataFrame.

    Args:
        path: Ruta del Parquet unificado.

    Returns:
        DataFrame con el esquema de `unify.UNIFIED_DATAFRAME_COLUMNS`.

    Raises:
        FileNotFoundError: Si `path` no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"Falta input para splits.py: {path}")
    df = pd.read_parquet(path)
    logger.info("splits.load_unified: %d filas leídas de %s", len(df), path)
    return df


def build_train_test_splits(
    unified: pd.DataFrame,
    *,
    kev_test_range: tuple[pd.Timestamp, pd.Timestamp] = KEV_TEST_RANGE,
    kev_freeze_date: pd.Timestamp = KEV_FREEZE_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deriva los DataFrames de train y test a partir del unificado.

    Train y test comparten cardinalidad y features (las del unified). La
    única diferencia es la columna de etiqueta:

    - `y_train = in_kev` (estado de KEV congelado a `kev_freeze_date`).
    - `y_test  = kev_date_added ∈ kev_test_range` (CVEs añadidos a KEV
      durante el periodo de evaluación).

    Args:
        unified: DataFrame producido por `unify.build_unified()`.
        kev_test_range: Tupla `(inicio, fin)` inclusiva que define el
            periodo de evaluación. Por defecto, todo 2024.
        kev_freeze_date: Fecha de corte de `in_kev` en el train. Usada
            para las assertions anti-leakage.

    Returns:
        Tupla `(train_df, test_df)` con el orden de filas del unified.

    Raises:
        ValueError: Si faltan columnas requeridas o si hay leakage
            temporal en alguno de los dos splits.
    """
    missing = [c for c in REQUIRED_UNIFIED_COLUMNS if c not in unified.columns]
    if missing:
        raise ValueError(
            f"Columnas requeridas ausentes en unified: {missing}"
        )

    test_start, test_end = kev_test_range

    train_df = unified.copy()
    train_df["y_train"] = unified["in_kev"].astype(bool)

    test_df = unified.copy()
    test_df["y_test"] = unified["kev_date_added"].between(
        test_start, test_end, inclusive="both"
    )
    # Ground truth secundario: KEV ∪ (primer exploit EDB en la ventana).
    edb_in_window = unified["edb_first_published"].between(
        test_start, test_end, inclusive="both"
    )
    test_df["y_test_multi"] = test_df["y_test"] | edb_in_window

    _assert_no_leakage(train_df, test_df, kev_freeze_date, test_start, test_end)

    logger.info(
        "splits.build: n=%d, train_pos=%d (%.2f%%), test_pos=%d (%.2f%%), "
        "test_multi_pos=%d (%.2f%%)",
        len(unified),
        int(train_df["y_train"].sum()),
        float(train_df["y_train"].mean()) * 100,
        int(test_df["y_test"].sum()),
        float(test_df["y_test"].mean()) * 100,
        int(test_df["y_test_multi"].sum()),
        float(test_df["y_test_multi"].mean()) * 100,
    )
    return train_df, test_df


def _assert_no_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kev_freeze_date: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> None:
    """Valida que ni train ni test tienen leakage temporal."""
    train_pos = train.loc[train["y_train"], "kev_date_added"]
    if train_pos.notna().any():
        max_train = train_pos.max()
        if pd.notna(max_train) and max_train > kev_freeze_date:
            raise ValueError(
                "Leakage en train: y_train=True con kev_date_added "
                f"{max_train} > freeze {kev_freeze_date.date()}"
            )

    test_pos = test.loc[test["y_test"], "kev_date_added"]
    if not test_pos.empty:
        min_test, max_test = test_pos.min(), test_pos.max()
        if min_test < test_start or max_test > test_end:
            raise ValueError(
                "Leakage en test: y_test=True con kev_date_added fuera del "
                f"rango [{test_start.date()}, {test_end.date()}] "
                f"(min={min_test}, max={max_test})"
            )

    # Positivos que entran solo por EDB: su primer exploit debe caer en la ventana.
    edb_only = test.loc[
        test["y_test_multi"] & ~test["y_test"], "edb_first_published"
    ]
    if not edb_only.empty:
        min_edb, max_edb = edb_only.min(), edb_only.max()
        if min_edb < test_start or max_edb > test_end:
            raise ValueError(
                "Leakage en test multi-source: y_test_multi=True por EDB con "
                f"edb_first_published fuera del rango [{test_start.date()}, "
                f"{test_end.date()}] (min={min_edb}, max={max_edb})"
            )


def save_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    train_path: Path = PROCESSED_TRAIN_PATH,
    test_path: Path = PROCESSED_TEST_PATH,
) -> tuple[Path, Path]:
    """Persiste los dos splits como Parquet snappy.

    Args:
        train: DataFrame del train (con `y_train`).
        test: DataFrame del test (con `y_test`).
        train_path: Ruta de salida del train.
        test_path: Ruta de salida del test.

    Returns:
        Tupla `(train_path, test_path)`.
    """
    for path, df, name in (
        (train_path, train, "train"),
        (test_path, test, "test"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression="snappy", index=False)
        logger.info(
            "splits.save: %s.parquet escrito en %s (%d filas)", name, path, len(df)
        )
    return train_path, test_path


def build_splits(
    *,
    unified_path: Path = PROCESSED_UNIFIED_PATH,
    train_path: Path = PROCESSED_TRAIN_PATH,
    test_path: Path = PROCESSED_TEST_PATH,
    kev_test_range: tuple[pd.Timestamp, pd.Timestamp] = KEV_TEST_RANGE,
) -> dict[str, int | float]:
    """Orquesta carga, derivación, persistencia y devuelve métricas.

    Args:
        unified_path: Ruta del `unified.parquet`.
        train_path: Ruta de salida del `train.parquet`.
        test_path: Ruta de salida del `test.parquet`.
        kev_test_range: Periodo de evaluación KEV (inclusivo).

    Returns:
        Dict con `n_rows`, `n_train_positives`, `n_test_positives`,
        `test_base_rate_pct` para que el orquestador del pipeline pueda
        loguearlo.
    """
    unified = load_unified(unified_path)
    train_df, test_df = build_train_test_splits(
        unified, kev_test_range=kev_test_range
    )
    save_splits(train_df, test_df, train_path=train_path, test_path=test_path)

    return {
        "n_rows": int(len(unified)),
        "n_train_positives": int(train_df["y_train"].sum()),
        "n_test_positives": int(test_df["y_test"].sum()),
        "n_test_multi_positives": int(test_df["y_test_multi"].sum()),
        "test_base_rate_pct": round(float(test_df["y_test"].mean()) * 100, 4),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stats = build_splits()
    logger.info("Smoke 9.1: %s", stats)
