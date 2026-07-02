"""Cliente KEV.

Descarga el catálogo CISA KEV (Known Exploited Vulnerabilities) desde el
feed público y lo persiste como Parquet. KEV actúa como ground truth del
experimento: un CVE entró en KEV ⇒ se está explotando activamente.
El unificador hará el left join NVD ← EPSS ← KEV usando este Parquet.

A diferencia de EPSS, KEV no tiene snapshots históricos públicos: el feed
es un único catálogo siempre vigente. Por eso aquí no se versiona por fecha;
el JSON crudo se sobrescribe en cada descarga.

Endpoint: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

KEV_JSON_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/"
    "known_exploited_vulnerabilities.json"
)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # 2, 4, 8 s
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

RAW_KEV_DIR = Path("data/raw/kev")
RAW_KEV_FILENAME = "known_exploited_vulnerabilities.json"
PROCESSED_KEV_PATH = Path("data/processed/kev.parquet")

KEV_DATAFRAME_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "date_added",
    "due_date",
    "vendor_project",
    "product",
)

logger = logging.getLogger(__name__)


def _request_with_backoff(url: str) -> bytes:
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
                "KEV HTTP %s (intento %d/%d); aplicando backoff",
                status,
                attempt + 1,
                MAX_RETRIES,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "KEV red caída (%s) (intento %d/%d); aplicando backoff",
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES,
            )

        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE ** (attempt + 1))

    assert last_exc is not None
    raise last_exc


def _download_json(url: str, dest: Path) -> Path:
    """Descarga `url` (JSON) a `dest`. Sobrescribe si existe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = _request_with_backoff(url)
    dest.write_bytes(content)
    logger.info("KEV JSON descargado: %s (%d bytes)", dest, len(content))
    return dest


def _parse_kev_json(path: Path) -> pd.DataFrame:
    """Lee el JSON del catálogo KEV y devuelve un DataFrame con las 5 columnas.

    Estructura del JSON: `{"title": ..., "catalogVersion": ..., "vulnerabilities": [...]}`.
    Cada item trae campos en camelCase (`cveID`, `dateAdded`, `dueDate`,
    `vendorProject`, `product`, ...). Filtra a los 5 necesarios y castea
    fechas a `pd.Timestamp` para permitir el freeze temporal de `in_kev`
    (`kev_date_added <= KEV_FREEZE_DATE`).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["vulnerabilities"])
    df = df.rename(
        columns={
            "cveID": "cve_id",
            "dateAdded": "date_added",
            "dueDate": "due_date",
            "vendorProject": "vendor_project",
        }
    )
    df["date_added"] = pd.to_datetime(df["date_added"])
    df["due_date"] = pd.to_datetime(df["due_date"])
    df = df[list(KEV_DATAFRAME_COLUMNS)]
    return df


def fetch_kev(
    *,
    raw_dir: Path = RAW_KEV_DIR,
    save_raw: bool = True,
) -> pd.DataFrame:
    """Descarga (o reusa) el catálogo KEV y lo devuelve parseado.

    Idempotencia: si `save_raw=True` y el JSON ya existe en `raw_dir`, no se
    re-descarga (se parsea el existente). Para forzar refresh, borrar el
    archivo en `raw_dir` antes de llamar.

    Args:
        raw_dir: Directorio donde se persiste el JSON crudo.
        save_raw: Si `False`, descarga, parsea, y elimina el JSON tras el parsing.

    Returns:
        DataFrame con 5 columnas: `cve_id`, `date_added`, `due_date`,
        `vendor_project`, `product`.

    Raises:
        requests.HTTPError: Si CISA devuelve un 4xx no retryable.
    """
    json_path = raw_dir / RAW_KEV_FILENAME

    reused = save_raw and json_path.exists()
    if reused:
        logger.info("KEV: reusando JSON existente %s", json_path)
    else:
        _download_json(KEV_JSON_URL, json_path)

    df = _parse_kev_json(json_path)

    if not save_raw and not reused and json_path.exists():
        json_path.unlink()

    logger.info("KEV catálogo: %d filas parseadas", len(df))
    return df


def build_kev_parquet(
    *,
    output_path: Path = PROCESSED_KEV_PATH,
    raw_dir: Path = RAW_KEV_DIR,
) -> Path:
    """Orquesta la descarga y persistencia del catálogo KEV como Parquet.

    Args:
        output_path: Ruta de salida del Parquet procesado.
        raw_dir: Carpeta donde se persiste el JSON crudo.

    Returns:
        Ruta del Parquet escrito.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_kev(raw_dir=raw_dir, save_raw=True)
    df.to_parquet(output_path, compression="snappy")
    logger.info("KEV parquet escrito: %s (%d filas)", output_path, len(df))
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    path = build_kev_parquet()
    logger.info("Smoke 7.1: %s", path)
