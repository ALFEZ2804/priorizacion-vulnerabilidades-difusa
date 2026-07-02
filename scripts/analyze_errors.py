"""Análisis de errores del FIS: KEV que no priorizó.

El reverso de `find_divergences.py`: en vez de los aciertos del FIS frente a CVSS,
mira sus **falsos negativos**: los CVEs que entraron en KEV durante 2024-2025
(`y_test`) pero que el FIS hundió en el ranking. Es la materia prima de la sección
de **Limitaciones** de la memoria y la que explica por qué el FIS no lidera en
Precision@K.

Para cada positivo descompone la `priority` en sus dos señales intermedias
(`severity` de FIS-1, `threat` de FIS-2) leídas de `scored.parquet`, de modo que el
porqué de cada miss es legible: el patrón es severity alta pero threat baja por un
EPSS percentil bajo y sin subida (`epss_delta`≈0), que colapsa la prioridad al
bucket de empate ~4,0. El FIS hereda así el punto ciego de EPSS ante
vulnerabilidades locales / de escalada de privilegios / 0-days.

Reutiliza el universo de evaluación KEV de `src.evaluation.dataset` (mismo que la
comparativa); no entrena el RF.

Salida: `data/processed/fis_misses.csv` + resumen agregado y peores casos por consola.

Uso:
    python scripts/analyze_errors.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.evaluation.dataset import SCORED_PATH, load_evaluation_frame  # noqa: E402

logger = logging.getLogger("analyze_errors")

MISSES_CSV_PATH = _PROJECT_ROOT / "data" / "processed" / "fis_misses.csv"

# Bandas de rank_fis para la tabla de la memoria (1 = más prioritario).
_BANDS = [(1, 50), (51, 100), (101, 500), (501, 1000),
          (1001, 5000), (5001, 20000), (20001, None)]

_OUTPUT_COLUMNS = [
    "rank_fis", "cvss_v3_score", "epss_percentile", "epss_delta", "log_cpe",
    "severity", "threat", "priority_raw", "priority", "kev_date_added",
]


def analyze_errors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve `(positivos, universo)` con `rank_fis` y descomposición del FIS.

    `positivos` son los `y_test` ordenados de peor a mejor rango del FIS (los misses
    más graves primero); `universo` es el frame completo de evaluación con `rank_fis`.
    """
    ev = load_evaluation_frame(ground_truth="kev")
    decomp = pd.read_parquet(SCORED_PATH)[
        ["cve_id", "severity", "threat", "priority_raw"]
    ].set_index("cve_id")
    ev = ev.join(decomp)

    ev["rank_fis"] = ev["priority"].rank(ascending=False, method="min")
    pos = ev[ev["y_test"].astype(bool)].sort_values("rank_fis", ascending=False)
    return pos[_OUTPUT_COLUMNS], ev


def _band_table(pos: pd.DataFrame, n_universe: int) -> pd.DataFrame:
    """Reparto de los positivos por banda de `rank_fis`."""
    rows = []
    for lo, hi in _BANDS:
        top = hi if hi is not None else n_universe
        count = int(((pos["rank_fis"] >= lo) & (pos["rank_fis"] <= top)).sum())
        rows.append({"banda": f"{lo}-{top}", "positivos": count})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    pos, ev = analyze_errors()
    n_pos = len(pos)

    MISSES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    pos.to_csv(MISSES_CSV_PATH)
    logger.info("escrito %s (%d positivos KEV)", MISSES_CSV_PATH, n_pos)

    bandas = _band_table(pos, len(ev))
    fuera_top500 = int((pos["rank_fis"] > 500).sum())

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print(f"\n=== Reparto de los {n_pos} positivos KEV por banda de rank_fis ===")
    print(bandas.to_string(index=False))
    print(f"\nFuera del top-500: {fuera_top500}/{n_pos} "
          f"({100 * fuera_top500 / n_pos:.0f} %)")

    print("\n=== Diagnóstico agregado de los positivos ===")
    print(f"  priority NaN (sin EPSS) : {int(pos['priority'].isna().sum())}")
    print(f"  cvss_v3_score < 7.0     : {int((pos['cvss_v3_score'] < 7.0).sum())}")
    print(f"  epss_percentile < 0.5   : {int((pos['epss_percentile'] < 0.5).sum())}")
    print(f"  log_cpe <= 0.7 (1 prod) : {int((pos['log_cpe'] <= 0.7).sum())}")
    print(f"  severity mediana        : {pos['severity'].median():.2f}")
    print(f"  threat   mediana        : {pos['threat'].median():.2f}")

    num_cols = pos.select_dtypes("number").columns
    shown = pos.copy()
    shown[num_cols] = shown[num_cols].round(3)
    print("\n=== 12 positivos KEV peor rankeados por el FIS (misses más graves) ===")
    print(shown.head(12).to_string())


if __name__ == "__main__":
    main()
