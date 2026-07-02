"""Universo de evaluación y scores de los 5 métodos.

Centraliza el montaje del conjunto de evaluación, compartido por la comparativa
(`scripts/run_evaluation.py`) y por el análisis de significancia
(`scripts/run_significance.py`), para no duplicar la definición del universo ni la
construcción de los scores entre ambos.

El universo son los CVEs que **no** estaban ya confirmados como explotables a la
fecha de corte (2023-12-31): la tarea es predecir cuáles entran en el ground truth
durante 2024-2025. Incluir los ya conocidos contaminaría el ranking (el `+2` del FIS
los pondría a todos en cabeza) y no tiene sentido operativo.

Dos ground truths:

- ``"kev"`` (principal): positivo = `y_test` (entró en KEV en 2024-2025); universo =
  no-`in_kev` a 2023-12-31.
- ``"multi"`` (secundario, multi-source): positivo = `y_test_multi` (KEV ∪ primer
  exploit EDB en 2024-2025); universo = no-`in_kev` **y** sin exploit EDB conocido
  antes del freeze (`edb_first_published ≤ 2023-12-31`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.baselines.combined import combined_score
from src.baselines.cvss import cvss_score
from src.baselines.epss import epss_score
from src.baselines.random_forest import rf_score, save_model, train_random_forest

SCORED_PATH = Path("data/processed/scored.parquet")
TEST_PATH = Path("data/processed/test.parquet")
TRAIN_PATH = Path("data/processed/train.parquet")

KEV_FREEZE_DATE = pd.Timestamp("2023-12-31")

GroundTruth = Literal["kev", "multi"]
_TARGET_COLUMN: dict[GroundTruth, str] = {"kev": "y_test", "multi": "y_test_multi"}

logger = logging.getLogger(__name__)


def load_evaluation_frame(*, ground_truth: GroundTruth = "kev") -> pd.DataFrame:
    """Carga el universo de evaluación con la `priority` del FIS.

    Args:
        ground_truth: ``"kev"`` (principal) o ``"multi"`` (multi-source). Determina
            qué CVEs ya-conocidos se excluyen del universo.

    Returns:
        DataFrame indexado por `cve_id` con las columnas del test más `priority`.
    """
    scored = pd.read_parquet(SCORED_PATH)[["cve_id", "priority"]]
    test = pd.read_parquet(TEST_PATH)
    ev = test.merge(scored, on="cve_id", how="left").set_index("cve_id")
    n_total = len(ev)

    known = ev["in_kev"].astype(bool)
    if ground_truth == "multi":
        edb_pre_freeze = ev["edb_first_published"].le(KEV_FREEZE_DATE).fillna(False)
        known = known | edb_pre_freeze
    ev = ev[~known].copy()

    target = _TARGET_COLUMN[ground_truth]
    logger.info(
        "universo de evaluación [%s]: %d candidatos (de %d) | positivos %s=%d",
        ground_truth, len(ev), n_total, target, int(ev[target].sum()),
    )
    return ev


def train_rf_model(*, save: bool = False):
    """Entrena el Random Forest sobre `train.parquet` (`y_train` = KEV).

    Aislado para poder entrenar **una sola vez** y reutilizar el mismo modelo en las
    evaluaciones KEV y multi-source (el RF no depende del ground truth de test).

    Args:
        save: Si `True`, persiste el modelo con `save_model`.

    Returns:
        El `Pipeline` (imputer + RF) ajustado.
    """
    logger.info("entrenando Random Forest...")
    rf_model = train_random_forest(pd.read_parquet(TRAIN_PATH))
    if save:
        save_model(rf_model)
    return rf_model


def build_method_scores(
    ev: pd.DataFrame | None = None,
    *,
    ground_truth: GroundTruth = "kev",
    rf_model=None,
    save_rf: bool = False,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Construye el dict de scores de los 5 métodos para un ground truth.

    Args:
        ev: Universo de evaluación; si es `None` se carga con `load_evaluation_frame`
            para el `ground_truth` indicado.
        ground_truth: ``"kev"`` (principal) o ``"multi"`` (secundario). El RF se
            entrena igual (`y_train` = KEV) en ambos casos; solo cambia el universo
            y el target devuelto.
        rf_model: Modelo RF ya entrenado a reutilizar. Si es `None`, se entrena con
            `train_rf_model`.
        save_rf: Si `True` y se entrena aquí, persiste el modelo.

    Returns:
        Tupla `(scores, y)` donde `scores` mapea
        `{"FIS", "CVSS", "EPSS", "Combinado", "RF"}` a su Serie de scores, e `y` es
        el ground truth binario (`y_test` o `y_test_multi`) alineado con `ev`.
    """
    if ev is None:
        ev = load_evaluation_frame(ground_truth=ground_truth)
    if rf_model is None:
        rf_model = train_rf_model(save=save_rf)
    scores = {
        "FIS": ev["priority"],
        "CVSS": cvss_score(ev),
        "EPSS": epss_score(ev),
        "Combinado": combined_score(ev),
        "RF": rf_score(rf_model, ev),
    }
    return scores, ev[_TARGET_COLUMN[ground_truth]]
