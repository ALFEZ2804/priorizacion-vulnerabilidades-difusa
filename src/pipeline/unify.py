"""Unificador de fuentes.

Consolida `nvd.parquet`, los snapshots EPSS 2023-12-31 y 2024-12-31, y
`kev.parquet` en un único `dataset.csv` (más `unified.parquet` legacy
intacto) que sirve de base para EDA, FIS jerárquico,
baselines y splits temporales.

Decisiones de diseño:

- EPSS: dos snapshots. El 2023-12-31 aporta `epss_percentile` (input
  congelado al cierre del train). El 2024-12-31 se usa solo para derivar
  `epss_delta = epss_2024 - epss_2023` como señal de volatilidad temporal
  del EPSS: input del FIS-2 "Amenaza".
- KEV: `in_kev` se congela a 2023-12-31 (`KEV_FREEZE_DATE`) para evitar
  que el train vea entradas KEV de 2024 (no-leakage temporal). KEV no
  entra como input al FIS; se aplica como post-procesado +2 al output
  final (clip [0, 10]).
- Inputs continuos del FIS (5): `exploitability_subscore`,
  `impact_subscore`, `epss_percentile`, `epss_delta`, `log_cpe`. Los
  componentes CVSS categóricos se descartaron por producir activaciones
  crisp.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.exploitdb import PROCESSED_EXPLOITDB_PATH

PROCESSED_NVD_PATH = Path("data/processed/nvd.parquet")
PROCESSED_EPSS_PATH = Path("data/processed/epss_2023-12-31.parquet")
PROCESSED_EPSS_2024_PATH = Path("data/processed/epss_2024-12-31.parquet")
PROCESSED_KEV_PATH = Path("data/processed/kev.parquet")
PROCESSED_UNIFIED_PATH = Path("data/processed/unified.parquet")
PROCESSED_DATASET_CSV_PATH = Path("data/processed/dataset.csv")

KEV_FREEZE_DATE = pd.Timestamp("2023-12-31")

UNIFIED_DATAFRAME_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "published",
    "exploitability_subscore",
    "impact_subscore",
    "epss_percentile",
    "epss_delta",
    "log_cpe",
    "cvss_v3_score",
    "in_kev",
    "kev_date_added",
    "edb_first_published",
    "target_year",
)

logger = logging.getLogger(__name__)


def load_sources(
    *,
    nvd_path: Path = PROCESSED_NVD_PATH,
    epss_path: Path = PROCESSED_EPSS_PATH,
    epss_2024_path: Path = PROCESSED_EPSS_2024_PATH,
    kev_path: Path = PROCESSED_KEV_PATH,
    exploitdb_path: Path = PROCESSED_EXPLOITDB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee los cinco Parquets de entrada y devuelve los DataFrames.

    Args:
        nvd_path: Ruta del Parquet de NVD (6 columnas).
        epss_path: Snapshot EPSS de cierre de train (2023-12-31).
        epss_2024_path: Snapshot EPSS de cierre de test (2024-12-31),
            usado solo para derivar `epss_delta`.
        kev_path: Ruta del Parquet de KEV.
        exploitdb_path: Ruta del Parquet de ExploitDB (ground truth
            secundario).

    Returns:
        Tupla `(nvd, epss_2023, epss_2024, kev, exploitdb)`.

    Raises:
        FileNotFoundError: Si alguna de las rutas no existe.
    """
    for path in (nvd_path, epss_path, epss_2024_path, kev_path, exploitdb_path):
        if not path.exists():
            raise FileNotFoundError(f"Falta input para unify.py: {path}")

    nvd_df = pd.read_parquet(nvd_path)
    epss_df = pd.read_parquet(epss_path)
    epss_2024_df = pd.read_parquet(epss_2024_path)
    kev_df = pd.read_parquet(kev_path)
    exploitdb_df = pd.read_parquet(exploitdb_path)

    logger.info(
        "unify.load_sources: nvd=%d, epss_2023=%d, epss_2024=%d, kev=%d, edb=%d",
        len(nvd_df),
        len(epss_df),
        len(epss_2024_df),
        len(kev_df),
        len(exploitdb_df),
    )
    return nvd_df, epss_df, epss_2024_df, kev_df, exploitdb_df


def build_unified_dataset(
    nvd: pd.DataFrame,
    epss_2023: pd.DataFrame,
    epss_2024: pd.DataFrame,
    kev: pd.DataFrame,
    exploitdb: pd.DataFrame | None = None,
    *,
    kev_freeze_date: pd.Timestamp = KEV_FREEZE_DATE,
) -> pd.DataFrame:
    """Construye el dataset unificado con los 5 inputs continuos del FIS.

    Cuatro left joins por `cve_id` (NVD ← EPSS 2023 ← EPSS 2024 ← KEV ← EDB).
    Deriva `epss_delta`, `log_cpe`, `in_kev` y `target_year`.

    Args:
        nvd: DataFrame con el esquema de `nvd.parquet` (6 columnas).
        epss_2023: Snapshot EPSS 2023-12-31 con `cve_id`, `epss`,
            `epss_percentile`, `epss_date`.
        epss_2024: Snapshot EPSS 2024-12-31, mismo esquema; solo se usa
            su `epss` para `epss_delta`.
        kev: DataFrame con el esquema de `kev.parquet`
            (`cve_id`, `date_added`, `due_date`, `vendor_project`, `product`).
        exploitdb: DataFrame de ExploitDB con `cve_id` y `edb_first_published`
            (ground truth secundario). Si es `None`, la columna
            `edb_first_published` se rellena con `NaT` (modo legacy / tests).
        kev_freeze_date: Fecha de corte para `in_kev`. Por defecto el
            cierre del periodo de train (2023-12-31).

    Returns:
        DataFrame con las columnas de `UNIFIED_DATAFRAME_COLUMNS` en orden.
    """
    epss_2023_join = epss_2023[["cve_id", "epss", "epss_percentile"]].rename(
        columns={"epss": "epss_2023"}
    )
    epss_2024_join = epss_2024[["cve_id", "epss"]].rename(columns={"epss": "epss_2024"})
    kev_join = kev[["cve_id", "date_added"]].rename(
        columns={"date_added": "kev_date_added"}
    )

    df = nvd.merge(epss_2023_join, on="cve_id", how="left")
    df = df.merge(epss_2024_join, on="cve_id", how="left")
    df = df.merge(kev_join, on="cve_id", how="left")

    if exploitdb is not None and len(exploitdb):
        edb_join = exploitdb[["cve_id", "edb_first_published"]]
        df = df.merge(edb_join, on="cve_id", how="left")
    else:
        df["edb_first_published"] = pd.Series(
            pd.NaT, index=df.index, dtype="datetime64[ns]"
        )

    df["epss_delta"] = df["epss_2024"] - df["epss_2023"]
    df["log_cpe"] = np.log1p(df["n_cpe_total"])
    df["in_kev"] = df["kev_date_added"].notna() & (
        df["kev_date_added"] <= kev_freeze_date
    )
    df["target_year"] = df["published"].dt.year.astype("int16")

    _warn_on_quality_issues(df)

    df = df[list(UNIFIED_DATAFRAME_COLUMNS)]
    return df


def _warn_on_quality_issues(df: pd.DataFrame) -> None:
    """Registra warnings ante anomalías esperadas pero no fatales."""
    n_dupes = int(df["cve_id"].duplicated().sum())
    if n_dupes > 0:
        logger.warning("unify: %d cve_id duplicados tras el join", n_dupes)

    n_no_score = int(df["cvss_v3_score"].isna().sum())
    if n_no_score > 0:
        logger.warning(
            "unify: %d filas sin cvss_v3_score (no debería ocurrir, nvd.parquet ya filtra)",
            n_no_score,
        )

    n_no_subscore = int(
        df["exploitability_subscore"].isna().sum() + df["impact_subscore"].isna().sum()
    )
    if n_no_subscore > 0:
        logger.warning("unify: %d subscores CVSS nulos", n_no_subscore)

    if len(df) == 0:
        return

    epss_cov = float(df["epss_percentile"].notna().mean())
    delta_cov = float(df["epss_delta"].notna().mean())
    kev_cov = float(df["in_kev"].mean())
    log_cpe_min = float(df["log_cpe"].min())
    log_cpe_max = float(df["log_cpe"].max())
    logger.info(
        "unify: cobertura epss_percentile=%.1f%%, epss_delta=%.1f%%, in_kev=%.1f%%, "
        "log_cpe=[%.2f, %.2f] (n=%d)",
        epss_cov * 100,
        delta_cov * 100,
        kev_cov * 100,
        log_cpe_min,
        log_cpe_max,
        len(df),
    )


def save_unified(df: pd.DataFrame, path: Path = PROCESSED_UNIFIED_PATH) -> Path:
    """Persiste el DataFrame unificado como Parquet (snappy).

    Args:
        df: DataFrame producido por `build_unified_dataset`.
        path: Ruta de salida.

    Returns:
        Ruta del Parquet escrito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)
    logger.info("unify: parquet escrito en %s (%d filas)", path, len(df))
    return path


def save_dataset_csv(df: pd.DataFrame, path: Path = PROCESSED_DATASET_CSV_PATH) -> Path:
    """Persiste el DataFrame unificado como CSV para visualización directa.

    Formato compatible con Excel / VS Code. `published` y `kev_date_added`
    se serializan como ISO; el consumidor debe parsearlas con
    `pd.read_csv(..., parse_dates=["published", "kev_date_added"])`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("unify: csv escrito en %s (%d filas)", path, len(df))
    return path


def build_unified(
    *,
    nvd_path: Path = PROCESSED_NVD_PATH,
    epss_path: Path = PROCESSED_EPSS_PATH,
    epss_2024_path: Path = PROCESSED_EPSS_2024_PATH,
    kev_path: Path = PROCESSED_KEV_PATH,
    exploitdb_path: Path = PROCESSED_EXPLOITDB_PATH,
    output_path: Path = PROCESSED_UNIFIED_PATH,
    csv_output_path: Path = PROCESSED_DATASET_CSV_PATH,
) -> dict[str, float | int]:
    """Orquesta carga, unificación, persistencia y devuelve métricas.

    Escribe dos artefactos:
    - `output_path` (Parquet): contrato interno del pipeline (splits, FIS).
    - `csv_output_path` (CSV): para visualización directa y entregables.

    Returns:
        Dict con `n_rows`, `epss_cov_pct`, `epss_delta_cov_pct`,
        `in_kev_count`.
    """
    nvd_df, epss_2023_df, epss_2024_df, kev_df, exploitdb_df = load_sources(
        nvd_path=nvd_path,
        epss_path=epss_path,
        epss_2024_path=epss_2024_path,
        kev_path=kev_path,
        exploitdb_path=exploitdb_path,
    )
    df = build_unified_dataset(
        nvd_df, epss_2023_df, epss_2024_df, kev_df, exploitdb_df
    )
    save_unified(df, output_path)
    save_dataset_csv(df, csv_output_path)

    return {
        "n_rows": int(len(df)),
        "epss_cov_pct": round(float(df["epss_percentile"].notna().mean()) * 100, 2)
        if len(df)
        else 0.0,
        "epss_delta_cov_pct": round(float(df["epss_delta"].notna().mean()) * 100, 2)
        if len(df)
        else 0.0,
        "in_kev_count": int(df["in_kev"].sum()),
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    stats = build_unified()
    logger.info("Smoke 7.2: %s", stats)
