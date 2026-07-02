"""Candidatos a caso de estudio: divergencias FIS vs CVSS.

Identifica los CVEs que el FIS prioriza alto (top-50 por `priority`) pero que CVSS
deprioriza, para documentarlos como casos de estudio en la memoria. Es el material
cualitativo que sostiene el argumento del TFG: el valor del FIS no es Precision@K
bruto, sino reordenar CVEs que CVSS infravalora porque integra EPSS, amenaza
temporal y blast-radius (`log_cpe`).

Reutiliza el montaje del universo de evaluación de `src.evaluation.dataset` (mismo
que `run_evaluation.py` / `run_significance.py`), así que el ranking es idéntico al
de la comparativa: universo = CVEs no confirmados como explotables a 2023-12-31,
positivos = entrada en KEV (y, como secundario, KEV ∪ ExploitDB) durante 2024-2025.
Aquí solo intervienen FIS (columna `priority`) y CVSS; no se entrena el RF.

Criterio de divergencia (FIS top-50 pero CVSS no):
    rank_fis <= 50  AND  rank_cvss > 50

Cada candidato se clasifica en `tipo_divergencia` para dar variedad a los casos:
- ``severidad_moderada``: `cvss_v3_score <= 7.5`. CVSS lo trata como severidad media
  pero el FIS lo eleva por EPSS / amenaza temporal / blast-radius.
- ``cvss_saturado``: `cvss_v3_score > 7.5`. CVSS lo marca crítico pero **no lo
  distingue** del montón (cientos empatan en 9.8/10.0); el FIS usa EPSS para
  reordenar dentro de la banda crítica.

Los candidatos se ordenan poniendo delante los **aciertos** del FIS (CVEs que CVSS
no priorizó arriba y que acabaron siendo explotados: `y_test` o `y_test_multi`), y
dentro de cada grupo por mayor distancia entre el rango CVSS y el rango FIS.

Salida: `data/processed/divergences.csv` + impresión del top ~15 por consola.

Uso:
    python scripts/find_divergences.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.baselines.cvss import cvss_score  # noqa: E402
from src.evaluation.dataset import load_evaluation_frame  # noqa: E402

logger = logging.getLogger("find_divergences")

DIVERGENCES_CSV_PATH = _PROJECT_ROOT / "data" / "processed" / "divergences.csv"

FIS_TOP_K = 50          # el FIS lo sitúa en cabeza
CVSS_TOP_K = 50         # CVSS NO lo sitúa en cabeza
CVSS_HIGH_THRESHOLD = 7.5   # frontera severidad moderada / banda crítica saturada

_OUTPUT_COLUMNS = [
    "tipo_divergencia",
    "priority", "rank_fis", "cvss_v3_score", "rank_cvss", "rank_gap",
    "epss_percentile", "epss_delta", "log_cpe",
    "entro_en_kev", "entro_multi", "kev_date_added", "edb_first_published",
    "published",
]


def find_divergences() -> pd.DataFrame:
    """Devuelve la tabla de candidatos a divergencia FIS>CVSS, ya ordenada."""
    ev = load_evaluation_frame(ground_truth="kev")

    # Solo FIS vs CVSS: el FIS es la columna `priority`, CVSS el baseline directo.
    # Rango 1 = más prioritario. method="min" => los empates comparten el mejor rango,
    # lo que refleja fielmente que CVSS no sabe ordenar dentro de un bucket de empate.
    rank_fis = ev["priority"].rank(ascending=False, method="min")
    rank_cvss = cvss_score(ev).rank(ascending=False, method="min")

    df = ev.copy()
    df["rank_fis"] = rank_fis
    df["rank_cvss"] = rank_cvss
    df["rank_gap"] = df["rank_cvss"] - df["rank_fis"]
    df["entro_en_kev"] = df["y_test"].astype(bool)
    df["entro_multi"] = df["y_test_multi"].astype(bool)
    df["tipo_divergencia"] = df["cvss_v3_score"].le(CVSS_HIGH_THRESHOLD).map(
        {True: "severidad_moderada", False: "cvss_saturado"}
    )

    mask = (df["rank_fis"] <= FIS_TOP_K) & (df["rank_cvss"] > CVSS_TOP_K)
    cand = df[mask].copy()

    # Aciertos del FIS primero (positivos KEV, luego positivos multi), después por
    # mayor distancia entre el rango CVSS y el rango FIS.
    cand = cand.sort_values(
        by=["entro_en_kev", "entro_multi", "rank_gap"],
        ascending=[False, False, False],
    )
    return cand[_OUTPUT_COLUMNS]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cand = find_divergences()

    n_kev = int(cand["entro_en_kev"].sum())
    n_multi = int((cand["entro_multi"] & ~cand["entro_en_kev"]).sum())
    por_tipo = cand["tipo_divergencia"].value_counts().to_dict()
    logger.info(
        "candidatos a divergencia: %d (FIS top-%d & CVSS no top-%d) | por tipo=%s | "
        "aciertos KEV=%d, aciertos solo-EDB=%d",
        len(cand), FIS_TOP_K, CVSS_TOP_K, por_tipo, n_kev, n_multi,
    )

    DIVERGENCES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(DIVERGENCES_CSV_PATH)
    logger.info("escrito %s (%d filas)", DIVERGENCES_CSV_PATH, len(cand))

    # Redondear solo las columnas numéricas para no chocar con las de fecha.
    num_cols = cand.select_dtypes("number").columns
    shown = cand.copy()
    shown[num_cols] = shown[num_cols].round(3)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n=== Aciertos del FIS (entraron en KEV/EDB y CVSS no los priorizó) ===")
    print(shown[shown["entro_multi"]].to_string())
    print("\n=== Casos por severidad moderada (CVSS <= 7.5, FIS top-50) ===")
    print(shown[shown["tipo_divergencia"] == "severidad_moderada"].head(10).to_string())


if __name__ == "__main__":
    main()
