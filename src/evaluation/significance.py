"""Significancia estadística de la comparativa de métodos.

Con solo 74 positivos en `y_test`, las diferencias de la tabla comparativa
podrían ser ruido de muestreo. Este módulo las cuantifica por **bootstrap de
remuestreo** del conjunto de evaluación.

Idea metodológica
-----------------
Los scores de cada método son fijos por CVE (ya calculados). Lo que se remuestrea,
con reemplazo, son las **filas** del universo de evaluación. En cada uno de los `B`
resamples se recalculan las métricas de ranking para **todos** los métodos sobre la
*misma* muestra (emparejado), lo que permite:

1. **IC percentil** por (método, métrica, K): la incertidumbre de cada métrica.
2. **IC de la diferencia** FIS − baseline (criterio principal de significancia): si
   el IC 95 % de la diferencia emparejada no contiene 0, la diferencia es
   significativa. Se reporta además el p-value del test de Wilcoxon de rangos con
   signo sobre las diferencias (lo pide el enunciado de la tarea), con la cautela de
   que, sobre muestras bootstrap, su lectura es descriptiva, no un contraste
   frecuentista clásico.

Implementación
--------------
El remuestreo se evalúa en O(n) por (resample, método) sin re-ordenar ni invocar a
`sklearn`: se pre-calcula el orden global descendente de cada método (idéntico al de
`metrics.py`, `argsort` estable con los NaN al fondo) y, por resample, se cuentan las
repeticiones de cada fila (`np.bincount`) y se acumulan sobre ese orden. Con esto el
bootstrap completo (B=2000, 5 métodos, 91.784 filas) corre en pocos segundos.

**Alcance:** el bootstrap cubre Precision@K y Recall@K (las métricas del criterio principal). NDCG@K se reporta como valor puntual en `results.csv`; no se
bootstrapea porque su descuento posicional no se vectoriza con el truco del multiset
y no es central para la decisión pendiente.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.evaluation.metrics import precision_at_k, recall_at_k

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (50, 100, 500)
BOOTSTRAP_METRICS: tuple[str, ...] = ("P", "R")


def _descending_order(scores: np.ndarray) -> np.ndarray:
    """Orden de índices por score descendente, NaN al fondo (igual que `metrics.py`)."""
    filled = scores.astype(float).copy()
    nan_mask = np.isnan(filled)
    if nan_mask.any():
        floor = np.nanmin(filled) - 1.0 if not nan_mask.all() else 0.0
        filled[nan_mask] = floor
    return np.argsort(-filled, kind="stable")


def _ranking_hits(y_ord: np.ndarray, c_ord: np.ndarray, k: int) -> float:
    """Positivos capturados en el top-K de un multiset.

    `y_ord` y `c_ord` son las etiquetas y los conteos de repetición de cada fila,
    ambos en orden de score descendente. K cuenta posiciones con multiplicidad; la
    fila que cruza la frontera entra de forma parcial.
    """
    cum = np.cumsum(c_ord)
    cum_hits = np.cumsum(y_ord * c_ord)
    pos = int(np.searchsorted(cum, k, side="left"))
    if pos >= c_ord.shape[0]:  # K abarca todo el multiset (no ocurre con n >> K)
        return float(cum_hits[-1])
    prev_cum = cum[pos - 1] if pos > 0 else 0.0
    prev_hits = cum_hits[pos - 1] if pos > 0 else 0.0
    remaining = k - prev_cum
    return float(prev_hits + y_ord[pos] * remaining)


def bootstrap_metric_distributions(
    scores: dict[str, pd.Series],
    y_true: pd.Series,
    ks: tuple[int, ...] = DEFAULT_KS,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[tuple[str, str, int], np.ndarray]:
    """Distribuciones bootstrap de Precision@K y Recall@K por método.

    Args:
        scores: Mapa método -> Serie de scores (mayor = más prioritario), alineable
            con `y_true`.
        y_true: Serie binaria de ground truth.
        ks: Cortes K a evaluar.
        n_boot: Número de remuestreos con reemplazo.
        seed: Semilla del generador (reproducibilidad bit-a-bit).

    Returns:
        Dict indexado por `(método, métrica, k)` con `métrica ∈ {"P", "R"}`, cada
        valor un `np.ndarray` de longitud `n_boot`. Todos los métodos comparten el
        mismo resample en cada iteración (emparejado).
    """
    y = pd.Series(y_true).astype(float)
    n = len(y)
    y_arr = y.to_numpy()

    orders = {name: _descending_order(pd.Series(s).reindex(y.index).to_numpy()) for name, s in scores.items()}
    y_ord = {name: y_arr[orders[name]] for name in scores}

    out: dict[tuple[str, str, int], np.ndarray] = {
        (name, metric, k): np.empty(n_boot)
        for name in scores
        for metric in BOOTSTRAP_METRICS
        for k in ks
    }

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        counts = np.bincount(idx, minlength=n).astype(float)
        total_pos = float((y_arr * counts).sum())
        for name in scores:
            c_ord = counts[orders[name]]
            yo = y_ord[name]
            for k in ks:
                hits = _ranking_hits(yo, c_ord, k)
                out[(name, "P", k)][b] = hits / k
                out[(name, "R", k)][b] = hits / total_pos if total_pos > 0 else 0.0
    return out


_POINT_FNS = {"P": precision_at_k, "R": recall_at_k}


def confidence_intervals(
    distributions: dict[tuple[str, str, int], np.ndarray],
    scores: dict[str, pd.Series],
    y_true: pd.Series,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Valor puntual e IC percentil `(1 - alpha)` por (método, métrica, K).

    El valor puntual se calcula con las funciones de `metrics.py` sobre los datos
    completos (autoridad: coincide con `results.csv`); los límites del IC son los
    percentiles de la distribución bootstrap.

    Returns:
        DataFrame con columnas `[method, metric, k, point, ci_lo, ci_hi]`.
    """
    y = pd.Series(y_true).astype(float)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for (name, metric, k), dist in distributions.items():
        s = pd.Series(scores[name]).reindex(y.index).to_numpy()
        point = _POINT_FNS[metric](s, y.to_numpy(), k)
        lo, hi = np.percentile(dist, [lo_q, hi_q])
        rows.append(
            {"method": name, "metric": metric, "k": k,
             "point": point, "ci_lo": float(lo), "ci_hi": float(hi)}
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["metric", "k", "method"]).reset_index(drop=True)


def paired_wilcoxon(
    distributions: dict[tuple[str, str, int], np.ndarray],
    *,
    reference: str = "FIS",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Contraste emparejado del método de referencia frente a cada baseline.

    Para cada (métrica, K) y cada baseline `≠ reference`, sobre las diferencias
    emparejadas `dist[reference] - dist[baseline]`:

    - `diff_median` y su IC percentil; `significant` = el IC no contiene 0.
    - `wilcoxon_p`: p-value bilateral del test de rangos con signo. Si todas las
      diferencias son nulas (p. ej. ambos métodos capturan 0 positivos en todo
      resample), el test es indefinido y se reporta `wilcoxon_p = 1.0`.

    Returns:
        DataFrame con `[metric, k, baseline, diff_median, diff_ci_lo, diff_ci_hi,
        significant, wilcoxon_p]`.
    """
    methods = sorted({name for name, _, _ in distributions})
    metric_ks = sorted({(m, k) for _, m, k in distributions})
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for metric, k in metric_ks:
        ref = distributions[(reference, metric, k)]
        for name in methods:
            if name == reference:
                continue
            diffs = ref - distributions[(name, metric, k)]
            lo, hi = np.percentile(diffs, [lo_q, hi_q])
            if np.allclose(diffs, 0.0):
                p = 1.0
            else:
                try:
                    p = float(wilcoxon(diffs).pvalue)
                except ValueError:
                    p = 1.0
            rows.append(
                {"metric": metric, "k": k, "baseline": name,
                 "diff_median": float(np.median(diffs)),
                 "diff_ci_lo": float(lo), "diff_ci_hi": float(hi),
                 "significant": not (lo <= 0.0 <= hi),
                 "wilcoxon_p": p}
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["metric", "k", "baseline"]).reset_index(drop=True)


def overlaps(ci_df: pd.DataFrame, a: str, b: str, metric: str, k: int) -> bool:
    """¿Solapan los IC de los métodos `a` y `b` para una (métrica, K)?

    Sobre la elección del ground truth: si para K ≤ 50 los IC del FIS y de los baselines
    fuertes no solapan, las conclusiones con KEV-solo son robustas.
    """
    def _bounds(method: str) -> tuple[float, float]:
        row = ci_df[(ci_df["method"] == method) & (ci_df["metric"] == metric) & (ci_df["k"] == k)]
        if row.empty:
            raise KeyError(f"sin IC para {method}/{metric}@{k}")
        return float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0])

    a_lo, a_hi = _bounds(a)
    b_lo, b_hi = _bounds(b)
    return a_lo <= b_hi and b_lo <= a_hi
