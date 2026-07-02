"""Evaluación FIS vs baselines sobre dos ground truths.

Rankea las 92.350 vulnerabilidades por cada método y mide cuántas de las que
entraron en el ground truth durante 2024-2025 caen en el top-K. El score del FIS es
la columna `priority` de `scored.parquet`; CVSS/EPSS/combinado se calculan sobre
`test.parquet`; el Random Forest se entrena sobre `train.parquet` (`y_train` = KEV)
y predice sobre el test. Sin leakage: `in_kev` (usado por el +2 del FIS y como
`y_train` del RF) está congelado a 2023-12-31, y los positivos son posteriores.

Dos tablas: `results.csv` con ground truth **KEV** (principal) y
`results_multisource.csv` con **KEV ∪ ExploitDB** (secundario, más positivos). El
montaje del universo y los scores vive en `src.evaluation.dataset` (compartido con
`scripts/run_significance.py`); el RF se entrena una sola vez y se reutiliza.

Uso:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.evaluation.compare import RESULTS_CSV_PATH, build_comparison_table, save_results  # noqa: E402
from src.evaluation.dataset import build_method_scores, train_rf_model  # noqa: E402

logger = logging.getLogger("run_evaluation")

RESULTS_MULTISOURCE_PATH = RESULTS_CSV_PATH.with_name("results_multisource.csv")

_GROUND_TRUTHS = (
    ("kev", "KEV 2024-2025", RESULTS_CSV_PATH),
    ("multi", "KEV + ExploitDB 2024-2025", RESULTS_MULTISOURCE_PATH),
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    rf_model = train_rf_model(save=True)  # entrenado una vez, reutilizado en ambos GT

    pd.set_option("display.width", 160)
    for ground_truth, label, path in _GROUND_TRUTHS:
        scores, y = build_method_scores(ground_truth=ground_truth, rf_model=rf_model)
        table = build_comparison_table(scores, y)
        print(f"\n=== Comparativa FIS vs baselines (ground truth: {label}) ===")
        print(table.round(4).to_string())
        save_results(table, path)


if __name__ == "__main__":
    main()
