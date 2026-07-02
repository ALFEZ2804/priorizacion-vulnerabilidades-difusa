"""Orquestador del pipeline de datos del TFG.

Reproduce de extremo a extremo la construcción del dataset:

    raw NVD JSON ──► nvd.parquet ──────┐
                                       │
    EPSS bulk CSV ──► epss_*.parquet ──┼──► unified.parquet ──► train.parquet
                                       │                    └─► test.parquet
    KEV JSON ──► kev.parquet ──────────┘

Reproducibilidad: borrar `data/processed/` y ejecutar este script con
`--skip-nvd-fetch` regenera el dataset entero a partir de `data/raw/`.
EPSS y KEV son idempotentes (reusan el raw cacheado si existe, descargan
si no) por lo que no necesitan flag de skip.

Uso típico (los JSON crudos de NVD ya están en disco):

    python scripts/build_dataset.py --skip-nvd-fetch

Para reconstruir todo desde la red:

    python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Permite ejecutar el script directamente desde la raíz sin instalar el proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.pipeline import epss, exploitdb, kev, nvd, splits, unify  # noqa: E402

logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-nvd-fetch",
        action="store_true",
        help="No re-descarga NVD. Asume data/raw/nvd/ poblado (default actual).",
    )
    parser.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        default=[2020, 2023],
        metavar=("YEAR_START", "YEAR_END"),
        help="Intervalo cerrado de años a conservar en nvd.parquet.",
    )
    parser.add_argument(
        "--epss-snapshot-dates",
        nargs="+",
        default=["2023-12-31", "2024-12-31"],
        help="Snapshots EPSS a descargar (formato ISO).",
    )
    return parser.parse_args()


def _step_nvd_fetch(year_range: tuple[int, int]) -> None:
    start = date(year_range[0], 1, 1)
    end = date(year_range[1], 12, 31)
    logger.info("Paso 5.2: descargando NVD entre %s y %s (puede tardar)", start, end)
    list(nvd.fetch_cves_in_range(start, end))


def _step_nvd_to_parquet(year_range: tuple[int, int]) -> Path:
    logger.info("Paso 6.1: consolidando NVD JSON → Parquet")
    _, stats = nvd.nvd_to_dataframe(year_range=tuple(year_range))
    logger.info("NVD stats: %s", stats)
    return nvd.PROCESSED_NVD_PATH


def _step_epss(snapshot_dates: list[date]) -> dict[date, Path]:
    logger.info("Paso 6.2: descargando snapshots EPSS %s", snapshot_dates)
    return epss.build_epss_parquets(snapshot_dates=snapshot_dates)


def _step_kev() -> Path:
    logger.info("Paso 7.1: descargando catálogo KEV")
    return kev.build_kev_parquet()


def _step_exploitdb() -> Path:
    logger.info("Parseando ExploitDB (ground truth secundario)")
    return exploitdb.build_exploitdb_parquet()


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    args = parse_args()
    year_range = tuple(args.year_range)
    snapshot_dates = [date.fromisoformat(s) for s in args.epss_snapshot_dates]

    if not args.skip_nvd_fetch:
        _step_nvd_fetch(year_range)
    else:
        logger.info("Saltando descarga NVD (--skip-nvd-fetch)")

    nvd_path = _step_nvd_to_parquet(year_range)
    logger.info("nvd.parquet listo en %s", nvd_path)

    epss_paths = _step_epss(snapshot_dates)
    for d, p in epss_paths.items():
        logger.info("epss[%s] listo en %s", d, p)

    kev_path = _step_kev()
    logger.info("kev.parquet listo en %s", kev_path)

    edb_path = _step_exploitdb()
    logger.info("exploitdb.parquet listo en %s", edb_path)

    logger.info("Paso 7.2: unificando NVD + EPSS + KEV + EDB")
    unified_stats = unify.build_unified()
    logger.info("unified.parquet listo en %s (%s)", unify.PROCESSED_UNIFIED_PATH, unified_stats)

    logger.info("Paso 9.1: construyendo splits temporales train/test")
    splits_stats = splits.build_splits()
    logger.info(
        "splits listos: train=%s, test=%s (%s)",
        splits.PROCESSED_TRAIN_PATH,
        splits.PROCESSED_TEST_PATH,
        splits_stats,
    )

    logger.info("Pipeline de datos completado.")


if __name__ == "__main__":
    main()
