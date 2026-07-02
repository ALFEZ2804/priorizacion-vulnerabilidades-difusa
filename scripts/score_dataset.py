"""Puntúa el dataset unificado con el FIS jerárquico.

Carga `data/processed/unified.parquet`, aplica `compute_priority` (los tres
sub-FIS encadenados + post-procesado KEV) y escribe
`data/processed/scored.parquet`: las 12 columnas del unified (incluida
`edb_first_published`) más `severity`, `threat`, `priority_raw` y
`priority`.

Uso:
    python scripts/score_dataset.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Permite ejecutar el script directamente desde la raíz sin instalar el proyecto.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.fis import compute_priority  # noqa: E402
from src.pipeline import unify  # noqa: E402

logger = logging.getLogger("score_dataset")

PROCESSED_SCORED_PATH = unify.PROCESSED_UNIFIED_PATH.parent / "scored.parquet"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    df = pd.read_parquet(unify.PROCESSED_UNIFIED_PATH)
    logger.info("unified cargado: %d filas x %d columnas", len(df), df.shape[1])

    t0 = time.perf_counter()
    scored = compute_priority(df)
    elapsed = time.perf_counter() - t0
    logger.info("compute_priority: %.2f s sobre %d CVEs", elapsed, len(scored))

    PROCESSED_SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(PROCESSED_SCORED_PATH, index=False)
    logger.info(
        "scored.parquet escrito en %s (%d filas x %d columnas)",
        PROCESSED_SCORED_PATH,
        len(scored),
        scored.shape[1],
    )

    # Sanity checks de cierre.
    n_nan = int(scored["priority"].isna().sum())
    pmin, pmax = scored["priority"].min(), scored["priority"].max()
    m_kev = scored.loc[scored["in_kev"], "priority"].mean()
    m_nokev = scored.loc[~scored["in_kev"], "priority"].mean()
    logger.info(
        "priority en [%.2f, %.2f] | NaN=%d (esperado ~29 sin EPSS) | media in_kev=%.2f vs no-kev=%.2f",
        pmin, pmax, n_nan, m_kev, m_nokev,
    )

    cols = [
        c
        for c in ["cve_id", "priority", "priority_raw", "severity", "threat",
                  "cvss_v3_score", "epss_percentile", "in_kev"]
        if c in scored.columns
    ]
    top = scored.nlargest(15, "priority")[cols]
    logger.info("Top-15 por priority:\n%s", top.to_string(index=False))


if __name__ == "__main__":
    main()
