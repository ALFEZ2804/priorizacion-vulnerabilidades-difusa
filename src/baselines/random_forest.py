"""Baseline Random Forest.

Clasificador de ML entrenado **apples-to-apples con el FIS**: usa exactamente las
mismas 5 señales de entrada que el FIS (sin `cvss_v3_score`, que el FIS tampoco usa
como entrada) y el mismo split temporal sin leakage. Representa la
alternativa de aprendizaje automático (más precisa pero opaca) frente al FIS
interpretable. Se entrena sobre `train.parquet` (target `y_train` = `in_kev`
congelado a 2023-12-31) y se evalúa sobre `test.parquet` (`y_test` = KEV 2024-2025).

El score de ranking es `predict_proba(...)[:, 1]`: la probabilidad estimada de que
el CVE acabe siendo explotado.

Notas metodológicas:
- Las 29 filas sin EPSS se imputan con la mediana del train (sklearn RF no acepta
  NaN). Es el tratamiento "estándar" cuya fragilidad contrasta el experimento de
  robustez frente a la degradación con gracia del FIS.
- `epss_delta` usa el snapshot EPSS de 2024-12-31 (posterior al corte), igual que
  el FIS; no revela el target (que EPSS varíe ≠ entrar en KEV). Limitación conocida
  y consistente entre ambos modelos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import ndcg_at_k

FEATURES: list[str] = [
    "exploitability_subscore",
    "impact_subscore",
    "epss_percentile",
    "epss_delta",
    "log_cpe",
]
TARGET_TRAIN = "y_train"
ORDER_COLUMN = "target_year"
MODEL_PATH = Path("data/processed/rf_model.joblib")

K_NDCG = 100          # corte de la métrica de selección, idéntico al fitness del FIS
N_ITER_SEARCH = 100   # presupuesto de búsqueda, igualado al random search del FIS

_PARAM_DISTRIBUTIONS = {
    "rf__n_estimators": [200, 300, 400],
    "rf__max_depth": [10, 20, 30, None],
    "rf__min_samples_leaf": [1, 5, 20, 50],
    "rf__max_features": ["sqrt", 0.5, 1.0],
    "rf__min_samples_split": [2, 10],
}

logger = logging.getLogger(__name__)


def _ndcg_scorer(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """NDCG@100 de `predict_proba`: la misma métrica que maximiza el FIS.

    Selecciona los hiperparámetros optimizando exactamente la métrica con la que
    después se compara el RF contra el FIS (apples-to-apples), en vez de un proxy
    como PR-AUC. El score de ranking es `predict_proba(...)[:, 1]` = P(explotado).
    """
    proba = estimator.predict_proba(X)
    score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return ndcg_at_k(score, np.asarray(y, dtype=float), K_NDCG)


def train_random_forest(
    train_df: pd.DataFrame, *, random_state: int = 42, n_iter: int = N_ITER_SEARCH
) -> Pipeline:
    """Entrena un RF con imputación por mediana y búsqueda de hiperparámetros.

    La búsqueda es **simétrica con la calibración del FIS**: mismo presupuesto
    (`n_iter` configuraciones muestreadas al azar), misma métrica objetivo
    (NDCG@100) y misma validación temporal. El CV usa `TimeSeriesSplit` sobre el
    orden temporal (`target_year`) para estimar el NDCG@100 *out-of-fold* sin
    mezclar años; seleccionar out-of-fold evita que el RF (de alta capacidad) se
    sobreajuste al elegir hiperparámetros (el FIS no lo necesita por su baja
    capacidad). El espacio cubre los hiperparámetros que de verdad importan con la
    clase tan minoritaria (566 positivos / 92.350): regularización de las hojas
    (`min_samples_leaf`, `min_samples_split`), profundidad y `max_features`.

    Args:
        train_df: `train.parquet` con `FEATURES`, `y_train` y `target_year`.
        random_state: Semilla del RF y del muestreo de configuraciones.
        n_iter: Nº de configuraciones evaluadas (presupuesto de búsqueda).

    Returns:
        El `Pipeline` (imputer + RF) ya ajustado con los mejores hiperparámetros.
    """
    df = train_df.sort_values(ORDER_COLUMN).reset_index(drop=True)
    X = df[FEATURES]
    y = df[TARGET_TRAIN].astype(int)

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "rf",
                # n_jobs=-1: el paralelismo del bosque usa hilos (prefer="threads"),
                # sin spawnear procesos. Es deliberado: este .venv está montado sobre
                # anaconda y la ruta tiene caracteres no ASCII, lo que rompe el spawn
                # de procesos de loky (re-lanzaba el script entero bajo el intérprete
                # base, con dos intérpretes compitiendo). Por eso la búsqueda va en
                # serie (n_jobs=1) y la paralelización vive aquí, en hilos.
                RandomForestClassifier(
                    class_weight="balanced", random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )
    search = RandomizedSearchCV(
        pipe,
        _PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=_ndcg_scorer,
        cv=TimeSeriesSplit(n_splits=4),
        random_state=random_state,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X, y)
    logger.info(
        "RF entrenado | mejores params=%s | NDCG@%d CV=%.4f",
        search.best_params_,
        K_NDCG,
        search.best_score_,
    )
    return search.best_estimator_


def rf_score(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    """Score de ranking del RF: P(explotado) = `predict_proba(...)[:, 1]`.

    Args:
        model: Pipeline entrenado por `train_random_forest`.
        df: DataFrame con `FEATURES` (puede contener NaN; el imputer los maneja).

    Returns:
        Serie float en [0, 1] indexada como `df`.
    """
    proba = model.predict_proba(df[FEATURES])[:, 1]
    return pd.Series(proba, index=df.index, name="rf_score")


def save_model(model: Pipeline, path: Path = MODEL_PATH) -> Path:
    """Persiste el modelo con joblib. Devuelve la ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("modelo RF guardado en %s", path)
    return path


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Carga un modelo persistido."""
    return joblib.load(path)
