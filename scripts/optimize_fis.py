"""Optimiza los cortes de las MFs de EPSS (FIS-2) por búsqueda en calibración.

Random search: muestrea N configuraciones de las funciones de pertenencia del
antecedente `epss_percentile`, evalúa cada una sobre el conjunto de **calibración**
(train: `y_train` = in_kev a 2023-12-31, 566 positivos) y selecciona la de mejor
NDCG@100. La ganadora se valida out-of-time en **test** (`y_test` = KEV 2024-25).

Objetivo: que los cortes de las MFs no se fijen a mano (arbitrarios) sino por un
procedimiento reproducible y defendible ("se evalúan N configuraciones en calibración
y se elige la mejor"). NO modifica el FIS de producción: solo lee y escribe en
`data/processed/optim/`. Su integración, si la config generaliza, es un paso posterior.

Decisiones metodológicas:
  - Fitness sobre `priority_raw` (SIN el +2 de in_kev): en train, in_kev ES el target,
    y el +2 inflaría el fitness con la etiqueta (leakage). En test no afecta (el
    universo excluye in_kev).
  - Reglas y formas de FIS-1/FIS-3 fijas (justificadas por dominio). Solo se mueven los
    cortes del antecedente epss_percentile de FIS-2, con restricción de orden.

Uso:
    python scripts/optimize_fis.py [N]      # N = nº de configuraciones (default 100)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skfuzzy as fuzz  # noqa: E402
from skfuzzy import control as ctrl  # noqa: E402

from src.fis import FuzzyInference  # noqa: E402
from src.fis._universe import universe  # noqa: E402
from src.fis.priority import build_priority_fis  # noqa: E402
from src.fis.threat import (  # noqa: E402
    _build_threat_rules,
    build_epss_delta_antecedent,
    build_threat_consequent,
)
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("optimize_fis")

SEED = 42
K_FITNESS = 100          # métrica que se maximiza: NDCG@100 (premia la cabeza)
K_REPORT = (50, 100, 500)
SCORED = _PROJECT_ROOT / "data/processed/scored.parquet"
TRAIN = _PROJECT_ROOT / "data/processed/train.parquet"
TEST = _PROJECT_ROOT / "data/processed/test.parquet"
OUT_DIR = _PROJECT_ROOT / "data/processed/optim"

# Configuración vigente en src/fis/threat.py (para el sanity check).
CURRENT = {"a": 0.3, "b": 0.5, "c": 0.7, "d": 0.9, "high_tri": False}


def build_threat_variant(a: float, b: float, c: float, d: float, high_tri: bool) -> FuzzyInference:
    """FIS-2 con los cortes (a<b<c<d) de epss_percentile; delta/reglas sin cambios.

    low = trapmf[0,0,a,b]; medium = trapmf[a,b,c,d];
    high = trimf[d,1,1] (no satura) si high_tri, si no trapmf[c,d,1,1] (como el actual).
    """
    var = ctrl.Antecedent(universe(0.0, 1.0, 0.01), "epss_percentile")
    var["low"] = fuzz.trapmf(var.universe, [0.0, 0.0, a, b])
    var["medium"] = fuzz.trapmf(var.universe, [a, b, c, d])
    var["high"] = (
        fuzz.trimf(var.universe, [d, 1.0, 1.0]) if high_tri
        else fuzz.trapmf(var.universe, [c, d, 1.0, 1.0])
    )
    dlt = build_epss_delta_antecedent()
    thr = build_threat_consequent()
    return FuzzyInference([var, dlt], thr, _build_threat_rules(var, dlt, thr))


def sample_config(rng: np.random.Generator) -> dict:
    """Muestrea cortes admisibles (0 <= a < b < c < d <= 1) + forma de 'high'."""
    a, b, c, d = sorted(rng.uniform(0.0, 1.0, size=4))
    return {"a": a, "b": b, "c": c, "d": d, "high_tri": bool(rng.random() < 0.5)}


def priority_raw_for(cfg: dict, fis3: FuzzyInference,
                     pct: np.ndarray, dlt: np.ndarray,
                     sev: np.ndarray, logcpe: np.ndarray) -> np.ndarray:
    """threat (FIS-2 variante) -> priority_raw (FIS-3 fijo). severity se reutiliza."""
    thr = build_threat_variant(cfg["a"], cfg["b"], cfg["c"], cfg["d"], cfg["high_tri"]).infer_batch(
        epss_percentile=pct, epss_delta=dlt
    )
    return fis3.infer_batch(severity=sev, threat=thr, log_cpe=logcpe)


def metrics_block(score: np.ndarray, y: np.ndarray, prefix: str) -> dict:
    """P/R/NDCG@K para un score; claves con prefijo (train_/test_)."""
    out = {}
    for k in K_REPORT:
        out[f"{prefix}P@{k}"] = round(precision_at_k(score, y, k), 4)
        out[f"{prefix}R@{k}"] = round(recall_at_k(score, y, k), 4)
        out[f"{prefix}NDCG@{k}"] = round(ndcg_at_k(score, y, k), 4)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    # --- Datos (alineados por cve_id) ------------------------------------------
    scored = pd.read_parquet(SCORED).set_index("cve_id")
    train = pd.read_parquet(TRAIN).set_index("cve_id").reindex(scored.index)
    test = pd.read_parquet(TEST).set_index("cve_id").reindex(scored.index)

    sev = scored["severity"].to_numpy(dtype=float)
    logcpe = scored["log_cpe"].to_numpy(dtype=float)
    pct = scored["epss_percentile"].to_numpy(dtype=float)
    dlt = scored["epss_delta"].to_numpy(dtype=float)
    in_kev = scored["in_kev"].astype(int).to_numpy()

    y_train = train["y_train"].astype(float).to_numpy()           # calibración (universo completo)
    test_mask = ~test["in_kev"].astype(bool).to_numpy()           # universo test oficial
    y_test = test["y_test"].astype(float).to_numpy()[test_mask]
    epss_test = scored["epss_percentile"].to_numpy(dtype=float)[test_mask]

    fis3 = build_priority_fis()

    def eval_full(cfg: dict) -> tuple[float, dict]:
        """Devuelve (fitness train, métricas train+test) para una config."""
        praw = priority_raw_for(cfg, fis3, pct, dlt, sev, logcpe)
        fit = ndcg_at_k(praw, y_train, K_FITNESS)
        prio = np.clip(praw + in_kev * 2.0, 0.0, 10.0)            # +2 solo para la validación test
        m = metrics_block(praw, y_train, "train_")
        m.update(metrics_block(prio[test_mask], y_test, "test_"))
        return fit, m

    # --- Sanity check: la config actual reproduce results.csv ------------------
    logger.info("sanity check: evaluando la configuración vigente de threat.py ...")
    fit_cur, m_cur = eval_full(CURRENT)
    print(f"\n[SANITY] config actual {CURRENT}")
    print(f"  fitness NDCG@100(train) = {fit_cur:.4f}")
    print(f"  test R@500 = {m_cur['test_R@500']} (esperado 0.108)  |  test P@100 = {m_cur['test_P@100']}")
    if abs(m_cur["test_R@500"] - 0.1081) > 0.01:
        logger.warning("¡el sanity check NO reproduce results.csv! revisar el reensamblado antes de fiarse.")

    # --- Random search ---------------------------------------------------------
    rng = np.random.default_rng(SEED)
    log_rows, best = [], None
    t0 = time.perf_counter()
    for i in range(n):
        cfg = sample_config(rng)
        praw = priority_raw_for(cfg, fis3, pct, dlt, sev, logcpe)
        fit = ndcg_at_k(praw, y_train, K_FITNESS)
        log_rows.append({**cfg, "fitness": round(fit, 5)})
        if best is None or fit > best["fitness"]:
            best = {**cfg, "fitness": fit}
        if (i + 1) % 10 == 0:
            logger.info("evaluadas %d/%d configs | mejor NDCG@100=%.4f | %.0fs",
                        i + 1, n, best["fitness"], time.perf_counter() - t0)

    # --- Validación de la ganadora en test -------------------------------------
    _, m_best = eval_full(best)
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "search_log.csv", index=False)
    (OUT_DIR / "best_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    comp = pd.DataFrame([
        {"método": "FIS actual", **{k.replace("test_", ""): v for k, v in m_cur.items() if k.startswith("test_")}},
        {"método": "FIS optimizado", **{k.replace("test_", ""): v for k, v in m_best.items() if k.startswith("test_")}},
        {"método": "EPSS", **metrics_block(epss_test, y_test, "")},
    ]).set_index("método")
    comp.to_csv(OUT_DIR / "best_vs_baselines.csv")

    print("\n" + "=" * 72)
    print(f"MEJOR CONFIG (de {n}, semilla {SEED}): "
          f"a={best['a']:.3f} b={best['b']:.3f} c={best['c']:.3f} d={best['d']:.3f} "
          f"high={'triangular' if best['high_tri'] else 'trapezoidal'}")
    print(f"fitness NDCG@100(train) = {best['fitness']:.4f}  (config actual: {fit_cur:.4f})")
    print("=" * 72)
    print("VALIDACIÓN EN TEST (KEV 2024-25, métricas oficiales):")
    print(comp.to_string())
    print(f"\ntiempo búsqueda: {time.perf_counter() - t0:.0f}s | resultados en {OUT_DIR}")


if __name__ == "__main__":
    main()
