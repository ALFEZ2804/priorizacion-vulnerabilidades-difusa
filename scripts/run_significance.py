"""Significancia estadística de la comparativa.

Bootstrap de remuestreo del conjunto de evaluación para acompañar cada métrica de la
tabla comparativa con su IC 95 % y contrastar el FIS frente a cada baseline (IC de la
diferencia emparejada + test de Wilcoxon). Reutiliza el mismo universo y scores que
`scripts/run_evaluation.py` vía `src.evaluation.dataset`.

Se ejecuta para los dos ground truths (KEV principal y KEV ∪ ExploitDB
secundario), con el RF entrenado una sola vez. Escribe por ground truth:
    significance[_multisource].csv           -> IC por (método, métrica, K)
    significance[_multisource]_wilcoxon.csv  -> FIS vs cada baseline

Uso:
    python scripts/run_significance.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.baselines.random_forest import load_model  # noqa: E402
from src.evaluation.dataset import build_method_scores  # noqa: E402
from src.evaluation.significance import (  # noqa: E402
    DEFAULT_KS,
    bootstrap_metric_distributions,
    confidence_intervals,
    overlaps,
    paired_wilcoxon,
)

logger = logging.getLogger("run_significance")

PROCESSED_DIR = Path("data/processed")
N_BOOT = 2000
SEED = 42

_GROUND_TRUTHS = (
    ("kev", "KEV 2024-2025", "significance.csv", "significance_wilcoxon.csv"),
    ("multi", "KEV + ExploitDB 2024-2025",
     "significance_multisource.csv", "significance_multisource_wilcoxon.csv"),
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 180)

    # Se reutiliza el modelo ya persistido por run_evaluation (mismo random_state,
    # idéntico): evita un reentrenamiento caro y garantiza coherencia con results.csv.
    rf_model = load_model()

    for ground_truth, label, ci_name, wilc_name in _GROUND_TRUTHS:
        scores, y = build_method_scores(ground_truth=ground_truth, rf_model=rf_model)
        logger.info("bootstrap [%s]: B=%d, K=%s ...", ground_truth, N_BOOT, DEFAULT_KS)
        dists = bootstrap_metric_distributions(
            scores, y, ks=DEFAULT_KS, n_boot=N_BOOT, seed=SEED
        )
        ci = confidence_intervals(dists, scores, y)
        wilc = paired_wilcoxon(dists, reference="FIS")

        print(f"\n=== IC bootstrap 95 %: ground truth: {label} ===")
        print(ci.round(4).to_string(index=False))
        print(f"\n=== FIS vs baselines (diferencia + Wilcoxon): {label} ===")
        print(wilc.round(4).to_string(index=False))
        print(f"\n=== Solapamiento FIS vs {{EPSS, RF}}: {label} ===")
        for metric in ("P", "R"):
            for k in (50, 100):
                for other in ("EPSS", "RF"):
                    ov = overlaps(ci, "FIS", other, metric, k)
                    print(f"{metric}@{k:<3} FIS vs {other:<4}: "
                          f"{'SOLAPAN' if ov else 'NO solapan'}")

        ci.to_csv(PROCESSED_DIR / ci_name, index=False)
        wilc.to_csv(PROCESSED_DIR / wilc_name, index=False)
        logger.info("escritos %s y %s", ci_name, wilc_name)


if __name__ == "__main__":
    main()
