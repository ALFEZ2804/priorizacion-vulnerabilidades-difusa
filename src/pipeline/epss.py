"""Cliente EPSS.

Descarga snapshots históricos de EPSS desde los CSV diarios bulk
publicados por FIRST.org / Cyentia. La elección del bulk CSV frente al
endpoint chunked `?cve=` responde a varios motivos: evita 500+ requests, garantiza
cobertura total del universo EPSS de cada fecha y produce snapshots
inmutables y trazables.

Dos snapshots para el split temporal honesto:

- 2023-12-31 → input EPSS del baseline EPSS y del FIS-C en train.
- 2024-12-31 → input EPSS para evaluación (cierre temporal del test).

Endpoint: https://epss.cyentia.com/epss_scores-YYYY-MM-DD.csv.gz
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

EPSS_CSV_URL_TEMPLATE = "https://epss.cyentia.com/epss_scores-{date}.csv.gz"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # 2, 4, 8 s
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

RAW_EPSS_DIR = Path("data/raw/epss")
PROCESSED_EPSS_DIR = Path("data/processed")
SNAPSHOT_DATES: tuple[date, ...] = (date(2023, 12, 31), date(2024, 12, 31))

EPSS_DATAFRAME_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "epss",
    "epss_percentile",
    "epss_date",
)

logger = logging.getLogger(__name__)


def _csv_filename(snapshot_date: date) -> str:
    """Nombre canónico del CSV crudo descargado (preserva la fecha)."""
    return f"epss_scores-{snapshot_date.isoformat()}.csv.gz"


def _parquet_filename(snapshot_date: date) -> str:
    """Nombre canónico del Parquet procesado."""
    return f"epss_{snapshot_date.isoformat()}.parquet"


def _request_with_backoff_epss(url: str) -> bytes:
    """GET `url` con reintentos exponenciales.

    Reintenta ante `RETRYABLE_STATUS` y errores de red. Backoff 2, 4, 8 s.
    Los demás 4xx (incl. 404) se propagan inmediatamente.
    """
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.content
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS:
                raise
            last_exc = exc
            logger.warning(
                "EPSS HTTP %s (intento %d/%d); aplicando backoff",
                status,
                attempt + 1,
                MAX_RETRIES,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "EPSS red caída (%s) (intento %d/%d); aplicando backoff",
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES,
            )

        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE ** (attempt + 1))

    assert last_exc is not None
    raise last_exc


def _download_csv_gz(url: str, dest: Path) -> Path:
    """Descarga `url` (CSV gzipeado) a `dest`. Sobrescribe si existe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = _request_with_backoff_epss(url)
    dest.write_bytes(content)
    logger.info("EPSS CSV descargado: %s (%d bytes)", dest, len(content))
    return dest


def _parse_epss_csv(path: Path) -> pd.DataFrame:
    """Lee el CSV gzipeado y devuelve un DataFrame con las columnas EPSS.

    El CSV tiene una línea de metadatos al inicio (`#model_version:...,score_date:...`)
    que `pd.read_csv(comment="#")` ignora. Columnas crudas: `cve, epss, percentile`.
    Renombra a `cve_id, epss, epss_percentile` y castea ambos floats explícitamente
    (algunos snapshots antiguos los traen como string).
    """
    df = pd.read_csv(path, comment="#", compression="gzip")
    df = df.rename(columns={"cve": "cve_id", "percentile": "epss_percentile"})
    df["epss"] = df["epss"].astype(float)
    df["epss_percentile"] = df["epss_percentile"].astype(float)
    return df


def fetch_epss_snapshot(
    snapshot_date: date,
    *,
    raw_dir: Path = RAW_EPSS_DIR,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Descarga (o reusa) el snapshot EPSS de `snapshot_date` y lo devuelve parseado.

    Idempotencia: si `save_raw=True` y el CSV ya existe en `raw_dir`, no se
    re-descarga (se parsea el existente). Útil al re-ejecutar el pipeline.

    Args:
        snapshot_date: Fecha del snapshot histórico (corte temporal).
        raw_dir: Directorio donde se persiste el CSV crudo.
        save_raw: Si `False`, descarga, parsea, y elimina el CSV tras el parsing.

    Returns:
        DataFrame con 4 columnas: `cve_id`, `epss`, `epss_percentile`, `epss_date`.

    Raises:
        requests.HTTPError: Si Cyentia devuelve un 4xx no retryable (p. ej. 404
            cuando no existe snapshot para esa fecha).
    """
    url = EPSS_CSV_URL_TEMPLATE.format(date=snapshot_date.isoformat())
    csv_path = raw_dir / _csv_filename(snapshot_date)

    reused = save_raw and csv_path.exists()
    if reused:
        logger.info("EPSS: reusando CSV existente %s", csv_path)
    else:
        _download_csv_gz(url, csv_path)

    df = _parse_epss_csv(csv_path)
    df["epss_date"] = snapshot_date.isoformat()
    df = df[list(EPSS_DATAFRAME_COLUMNS)]

    if not save_raw and not reused and csv_path.exists():
        csv_path.unlink()

    logger.info(
        "EPSS snapshot %s: %d filas parseadas", snapshot_date.isoformat(), len(df)
    )
    return df


def build_epss_parquets(
    snapshot_dates: Iterable[date] = SNAPSHOT_DATES,
    *,
    output_dir: Path = PROCESSED_EPSS_DIR,
    raw_dir: Path = RAW_EPSS_DIR,
) -> dict[date, Path]:
    """Orquesta la descarga y persistencia de N snapshots EPSS.

    Genera un Parquet por snapshot (no un único archivo con `snapshot_date`):
    cada split del experimento usa exactamente uno, y el nombre del archivo
    actúa como trazabilidad pura.

    Args:
        snapshot_dates: Fechas de corte a descargar.
        output_dir: Carpeta destino de los Parquet procesados.
        raw_dir: Carpeta donde se persisten los CSV crudos.

    Returns:
        Dict `{snapshot_date: parquet_path}` con las rutas escritas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[date, Path] = {}

    for snapshot_date in snapshot_dates:
        df = fetch_epss_snapshot(snapshot_date, raw_dir=raw_dir, save_raw=True)
        parquet_path = output_dir / _parquet_filename(snapshot_date)
        df.to_parquet(parquet_path, compression="snappy")
        logger.info("EPSS parquet escrito: %s (%d filas)", parquet_path, len(df))
        written[snapshot_date] = parquet_path

    return written


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    paths = build_epss_parquets()
    for d, p in paths.items():
        logger.info("Smoke 6.2: %s → %s", d, p)
