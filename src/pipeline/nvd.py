"""Cliente NVD.

- `fetch_cve(cve_id)` descarga un CVE individual.
- `fetch_cves_in_range(start, end)` itera todos los CVEs publicados
  en un rango de fechas, con paginación, rate limiting y backoff.
- `nvd_to_dataframe(raw_dir)` consolida los JSON crudos persistidos
  por `fetch_cves_in_range` en un Parquet normalizado, filtrando CVEs sin CVSS v3 (los
  componentes v3 son input obligatorio del FIS: ver
  el diseño).

Las tres funciones normalizan al esquema plano de `unified.parquet`
reutilizando `_parse_cve`.

Endpoint: https://services.nvd.nist.gov/rest/json/cves/2.0
"""

import json
import logging
import os
import re
import time
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import requests

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
REQUEST_TIMEOUT = 20
CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")

RESULTS_PER_PAGE = 2000  # máximo permitido por NVD
REQUEST_DELAY = 0.7  # s entre requests (límite con key: 50 req/30s → 1 req/0.6s)
MAX_RETRIES = 5  # intentos ante 403/429/5xx/Timeout/ConnectionError
BACKOFF_BASE = 2.0  # 2, 4, 8, 16, 32 s
RAW_NVD_DIR = Path("data/raw/nvd")
PROCESSED_NVD_PATH = Path("data/processed/nvd.parquet")
RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}

NVD_DATAFRAME_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "published",
    "cvss_v3_score",
    "exploitability_subscore",
    "impact_subscore",
    "n_cpe_total",
)

logger = logging.getLogger(__name__)


def _extract_description(cve: dict) -> str | None:
    """Devuelve la descripción en inglés del CVE, o la primera disponible.

    Args:
        cve: Objeto `cve` del JSON de NVD.

    Returns:
        Texto de la descripción, o `None` si no hay descripciones.
    """
    descriptions = cve.get("descriptions", [])
    if not descriptions:
        return None
    for d in descriptions:
        if d.get("lang") == "en":
            return d.get("value")
    return descriptions[0].get("value")


def _get_cvss_metric(cve: dict) -> dict | None:
    """Devuelve el metric CVSS v3.x completo con preferencia v3.1 → v3.0.

    El metric incluye `cvssData` (con baseScore, vector, componentes) y
    `exploitabilityScore` / `impactScore` como hermanos. Devolvemos el
    item entero porque los subscores viven a nivel de metric, no dentro
    de `cvssData`.

    Si hay múltiples scoring sources en `cvssMetricV31` (NVD + vendor),
    se toma el primero (suele ser NVD). Si solo hay v2, devuelve `None`.
    """
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        items = metrics.get(key, [])
        if items:
            return items[0]
    return None


def _extract_cvss_fields(metric: dict | None) -> dict:
    """Aplana el metric CVSS v3.x en los 3 campos del nuevo esquema.

    Devuelve `cvss_v3_score`, `exploitability_subscore` e `impact_subscore`.
    Si entra `None`, los tres son `None`.
    """
    if metric is None:
        return {
            "cvss_v3_score": None,
            "exploitability_subscore": None,
            "impact_subscore": None,
        }
    cvss_data = metric.get("cvssData", {}) or {}
    return {
        "cvss_v3_score": cvss_data.get("baseScore"),
        "exploitability_subscore": metric.get("exploitabilityScore"),
        "impact_subscore": metric.get("impactScore"),
    }


def _parse_cve(cve: dict, *, warn_missing: bool = True) -> dict:
    """Compone el dict plano de salida a partir del objeto `cve` de NVD.

    Reutilizable sobre cada elemento de `vulnerabilities[*]`.
    Loggea warnings si faltan campos opcionales clave (descripción, CVSS v3).

    `warn_missing=False` silencia esos warnings: usado al convertir NVD a DataFrame
    cuando consolida ~92k CVEs en bulk, donde el conteo agregado del
    `stats_dict` ya cubre la información y los warnings se vuelven ruido.
    """
    metric = _get_cvss_metric(cve)

    if warn_missing:
        if _extract_description(cve) is None:
            logger.warning("CVE %s has no description", cve.get("id"))
        if metric is None:
            logger.warning("CVE %s has no CVSS v3 metrics", cve.get("id"))

    return {
        "cve_id": cve["id"],
        "published": cve["published"],
        **_extract_cvss_fields(metric),
        "n_cpe_total": len(cve.get("configurations", [])),
    }


def fetch_cve(cve_id: str) -> dict:
    """Descarga un CVE de la API v2 de NVD y devuelve un dict plano.

    Args:
        cve_id: Identificador del CVE en formato `CVE-YYYY-NNNN(N+)`.

    Returns:
        Dict con los 6 campos del esquema NVD: `cve_id`, `published`,
        `cvss_v3_score`, `exploitability_subscore`, `impact_subscore`,
        `n_cpe_total`. Los 3 campos CVSS valen `None` si el CVE solo
        tiene v2.

    Raises:
        ValueError: Si `cve_id` no respeta el formato `CVE-YYYY-NNNN(N+)`.
        requests.HTTPError: Si la API responde con 4xx o 5xx.
        requests.Timeout: Si el request supera `REQUEST_TIMEOUT` segundos.
        LookupError: Si la API no devuelve el CVE (lista `vulnerabilities` vacía).
    """
    if not CVE_ID_PATTERN.match(cve_id):
        raise ValueError(f"Invalid CVE ID format: {cve_id}")

    api_key = os.getenv("NVD_API_KEY")
    if not api_key:
        logger.warning(
            "NVD_API_KEY no configurada; se usa la API pública de NVD "
            "con un límite de peticiones menor (5 cada 30 s)"
        )

    logger.info("Fetching CVE %s from NVD", cve_id)

    response = requests.get(
        NVD_API_URL,
        headers={"apiKey": api_key} if api_key else {},
        params={"cveId": cve_id},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        raise LookupError(f"CVE not found in NVD: {cve_id}")

    cve = vulnerabilities[0]["cve"]

    if cve.get("id", "").upper() != cve_id.upper():
        logger.warning("Requested %s but got %s", cve_id, cve.get("id"))

    return _parse_cve(cve)


def _format_nvd_date(d: date, end_of_day: bool = False) -> str:
    """Formatea una fecha en el ISO 8601 que espera NVD: 'YYYY-MM-DDTHH:MM:SS.000'.

    La API exige milisegundos explícitos. `end_of_day=True` ancla a 23:59:59.999
    para que `pubEndDate` incluya el día completo.
    """
    t = dtime(23, 59, 59, 999000) if end_of_day else dtime(0, 0, 0, 0)
    return datetime.combine(d, t).strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}"


def _request_with_backoff(params: dict, api_key: str) -> dict:
    """GET a `NVD_API_URL` con reintentos exponenciales.

    Reintenta ante códigos de `RETRYABLE_STATUS` y errores de red
    (`Timeout`, `ConnectionError`). Backoff: 2, 4, 8, 16, 32 s.
    Los demás 4xx se propagan inmediatamente (son bugs nuestros, no transitorios).

    Raises:
        requests.HTTPError: si tras `MAX_RETRIES` sigue fallando o si el código no es retryable.
        requests.Timeout / requests.ConnectionError: si la red sigue caída tras los reintentos.
    """
    headers = {"apiKey": api_key} if api_key else {}
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                NVD_API_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS:
                raise
            last_exc = exc
            logger.warning(
                "NVD HTTP %s (intento %d/%d); aplicando backoff", status, attempt + 1, MAX_RETRIES
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            logger.warning(
                "NVD red caída (%s) (intento %d/%d); aplicando backoff",
                type(exc).__name__,
                attempt + 1,
                MAX_RETRIES,
            )

        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE ** (attempt + 1))

    assert last_exc is not None
    raise last_exc


def _save_raw_page(payload: dict, start: date, end: date, page_idx: int) -> Path:
    """Guarda el JSON crudo de una página en `RAW_NVD_DIR`.

    Naming: `cves_{start_iso}_{end_iso}_page{N}.json`
    (derivado del archivo existente en `data/raw/nvd/exploration/`).
    """
    RAW_NVD_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_NVD_DIR / f"cves_{start.isoformat()}_{end.isoformat()}_page{page_idx}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fetch_cves_in_range(
    start: date,
    end: date,
    *,
    save_raw: bool = True,
) -> Iterator[dict]:
    """Descarga todos los CVEs publicados entre `start` y `end` (inclusive).

    Itera páginas con `resultsPerPage=2000` hasta cubrir `totalResults`.
    Rate limiting con `time.sleep(REQUEST_DELAY)` entre requests. Reutiliza
    `_parse_cve` sobre cada `vulnerabilities[i]["cve"]`.

    Args:
        start: Fecha de inicio del rango (incluida desde las 00:00:00).
        end: Fecha de fin del rango (incluida hasta las 23:59:59.999).
        save_raw: Si `True`, persiste cada página como JSON crudo en
            `data/raw/nvd/`.

    Yields:
        Dict normalizado con el esquema NVD del `unified.parquet`.

    Raises:
        ValueError: si `start > end`.
        RuntimeError: si `totalResults` no coincide con los CVEs entregados.
        requests.HTTPError / requests.Timeout / requests.ConnectionError:
            si la API falla tras los reintentos.
    """
    if start > end:
        raise ValueError(f"start ({start}) debe ser <= end ({end})")

    api_key = os.getenv("NVD_API_KEY")
    if not api_key:
        logger.warning(
            "NVD_API_KEY no configurada; se usa la API pública de NVD "
            "con un límite de peticiones menor (5 cada 30 s)"
        )

    base_params = {
        "pubStartDate": _format_nvd_date(start),
        "pubEndDate": _format_nvd_date(end, end_of_day=True),
        "resultsPerPage": RESULTS_PER_PAGE,
    }

    start_index = 0
    page_idx = 0
    total_results: int | None = None
    downloaded = 0

    logger.info("NVD: descargando CVEs de %s a %s", start, end)

    while True:
        params = {**base_params, "startIndex": start_index}
        payload = _request_with_backoff(params, api_key)

        if total_results is None:
            total_results = int(payload.get("totalResults", 0))
            logger.info("NVD: totalResults=%d", total_results)
            if total_results == 0:
                break

        vulnerabilities = payload.get("vulnerabilities", [])
        page_count = len(vulnerabilities)

        if save_raw:
            _save_raw_page(payload, start, end, page_idx)

        for vuln in vulnerabilities:
            yield _parse_cve(vuln["cve"])

        downloaded += page_count
        logger.info(
            "NVD: página %d → %d CVEs (startIndex=%d, acumulado=%d/%d)",
            page_idx,
            page_count,
            start_index,
            downloaded,
            total_results,
        )

        start_index += RESULTS_PER_PAGE
        page_idx += 1

        if start_index >= total_results:
            break

        time.sleep(REQUEST_DELAY)

    if total_results is not None and downloaded != total_results:
        msg = f"NVD: descargados {downloaded} CVEs, esperados {total_results}"
        logger.error(msg)
        raise RuntimeError(msg)


def _iter_raw_json_files(raw_dir: Path) -> Iterator[Path]:
    """Itera, en orden lexicográfico, los JSON de páginas crudas en `raw_dir`.

    Glob no recursivo `cves_*.json` para no recoger ficheros del subdir
    `exploration/` que comparte el directorio. Orden determinista facilita
    tests reproducibles.
    """
    yield from sorted(raw_dir.glob("cves_*.json"))


def _iter_cves_from_files(files: Iterable[Path]) -> Iterator[dict]:
    """Yield cada objeto `vulnerabilities[i].cve` sin cargar todos los JSON en RAM.

    Cada archivo de página puede pesar 10-30 MB; iterar de uno en uno mantiene
    el pico de memoria controlado al consolidar las 35 páginas (~262 MB).
    """
    for path in files:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        for vuln in payload.get("vulnerabilities", []):
            cve = vuln.get("cve")
            if cve is not None:
                yield cve


def nvd_to_dataframe(
    raw_dir: Path = RAW_NVD_DIR,
    *,
    year_range: tuple[int, int] | None = (2020, 2023),
    output_path: Path | None = PROCESSED_NVD_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Consolida los JSON crudos de `raw_dir` en un DataFrame normalizado.

    Aplica dos filtros, en este orden, sobre los CVEs encontrados:
    1. `year_range`: descarta CVEs cuyo año de `published` cae fuera del
       intervalo cerrado. Default `(2020, 2023)` (universo ampliado
        para mejorar el poder estadístico del test set; el
       smoke de enero 2024 en `data/raw/nvd/` también queda excluido).
    2. CVSS v3 obligatorio: descarta CVEs sin `cvss_v3_score` (cubre
       RESERVED/REJECTED y CVEs solo-v2). Justificación en
       el diseño.

    El DataFrame final tiene los 6 campos de `NVD_DATAFRAME_COLUMNS`:
    `published` como `datetime64[ns]`, los 3 campos CVSS como `float64`
    y `n_cpe_total` como `int64`.

    Args:
        raw_dir: Directorio con las páginas crudas (`cves_*.json`).
        year_range: Intervalo cerrado de años a conservar. `None` desactiva el filtro.
        output_path: Si no es `None`, escribe el Parquet (compresión snappy).

    Returns:
        Tupla `(df, stats)` con `stats` = {`total_loaded`,
        `dropped_out_of_range`, `dropped_no_v3`, `kept`}.
    """
    stats = {
        "total_loaded": 0,
        "dropped_out_of_range": 0,
        "dropped_no_v3": 0,
        "kept": 0,
    }
    rows: list[dict] = []

    for cve in _iter_cves_from_files(_iter_raw_json_files(raw_dir)):
        stats["total_loaded"] += 1

        if year_range is not None:
            published = cve.get("published", "")
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            if year is None or not (year_range[0] <= year <= year_range[1]):
                stats["dropped_out_of_range"] += 1
                continue

        row = _parse_cve(cve, warn_missing=False)
        if row["cvss_v3_score"] is None:
            stats["dropped_no_v3"] += 1
            continue

        rows.append(row)
        stats["kept"] += 1

    logger.info("nvd_to_dataframe stats: %s", stats)

    df = pd.DataFrame(rows, columns=list(NVD_DATAFRAME_COLUMNS))
    df["published"] = pd.to_datetime(df["published"], errors="raise")

    n_missing_subscores = int(
        df["exploitability_subscore"].isna().sum() + df["impact_subscore"].isna().sum()
    )
    if n_missing_subscores > 0:
        logger.warning(
            "nvd_to_dataframe: %d subscores nulos pese a tener cvss_v3_score "
            "(metric sin exploitabilityScore/impactScore en la API)",
            n_missing_subscores,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression="snappy")
        logger.info("nvd_to_dataframe: parquet escrito en %s", output_path)

    return df, stats


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    cves = list(fetch_cves_in_range(date(2024, 1, 1), date(2024, 1, 31)))
    logger.info("Smoke test 5.2: descargados %d CVEs en enero 2024", len(cves))
