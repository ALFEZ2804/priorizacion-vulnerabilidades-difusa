"""Pre-requisito (b) de la integración de la config optimizada del FIS.

Compara el **FIS optimizado** (config ganadora de `optimize_fis.py`, MF de
`epss_percentile` recalibrada) contra los cinco métodos, en
particular el baseline **Combinado** (CVSS×EPSS), que lideraba en multi-source,
bajo los **dos ground truths** (KEV principal y KEV∪ExploitDB secundario).

Pregunta que responde: *con la config optimizada, ¿el Combinado sigue batiendo
al FIS en multi-source, o el FIS-optim lo alcanza/supera?* Es el último número
que faltaba antes de decidir la integración en `src/fis/threat.py`.

Aislado y no destructivo: lee `scored.parquet`/`train`/`test` y `optim/best_config.json`,
reutiliza el harness de evaluación de `src.evaluation.dataset` y las MFs variantes
de la lógica de `optimize_fis.py`. Solo escribe en `data/processed/optim/`. No toca
`src/`, ni `results.csv`, ni `scored.parquet`.

Uso:
    python scripts/qa_optim_multisource.py
"""

from __future__ import annotations

import json
import logging
import sys
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
from src.evaluation.dataset import (  # noqa: E402
    build_method_scores,
    load_evaluation_frame,
    train_rf_model,
)
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("qa_optim_multisource")

K_REPORT = (50, 100, 500)
SCORED = _PROJECT_ROOT / "data/processed/scored.parquet"
OPTIM_DIR = _PROJECT_ROOT / "data/processed/optim"
BEST_CONFIG = OPTIM_DIR / "best_config.json"


def build_threat_variant(a: float, b: float, c: float, d: float, high_tri: bool) -> FuzzyInference:
    """FIS-2 con los cortes (a<b<c<d) de epss_percentile; delta/reglas sin cambios.

    Réplica exacta de `optimize_fis.build_threat_variant` (no importable: `scripts`
    no es paquete). low=trapmf[0,0,a,b]; medium=trapmf[a,b,c,d];
    high=trimf[d,1,1] (no satura) si high_tri, si no trapmf[c,d,1,1].
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


def fis_optim_priority(cfg: dict) -> pd.Series:
    """`priority` del FIS-optim para todo `scored.parquet`, indexada por cve_id.

    Reusa `severity` y `log_cpe` ya calculados en `scored.parquet` (FIS-1 y la
    forma de FIS-3 no cambian); solo recomputa `threat` (FIS-2 variante) y la
    prioridad final. Aplica el post-proceso KEV (+2, clip [0,10]); en el universo
    de evaluación (in_kev=False) es inocuo, pero deja una `priority` coherente.
    """
    scored = pd.read_parquet(SCORED).set_index("cve_id")
    thr = build_threat_variant(cfg["a"], cfg["b"], cfg["c"], cfg["d"], cfg["high_tri"]).infer_batch(
        epss_percentile=scored["epss_percentile"].to_numpy(dtype=float),
        epss_delta=scored["epss_delta"].to_numpy(dtype=float),
    )
    praw = build_priority_fis().infer_batch(
        severity=scored["severity"].to_numpy(dtype=float),
        threat=thr,
        log_cpe=scored["log_cpe"].to_numpy(dtype=float),
    )
    prio = np.clip(praw + scored["in_kev"].astype(int).to_numpy() * 2.0, 0.0, 10.0)
    return pd.Series(prio, index=scored.index, name="FIS-optim")


def metrics_row(method: str, score, y) -> dict:
    """Fila P/R/NDCG@K para un método (score y y alineados por posición)."""
    row = {"método": method}
    for k in K_REPORT:
        row[f"P@{k}"] = round(precision_at_k(score, y, k), 4)
        row[f"R@{k}"] = round(recall_at_k(score, y, k), 4)
        row[f"NDCG@{k}"] = round(ndcg_at_k(score, y, k), 4)
    return row


def evaluate(ground_truth: str, fis_optim: pd.Series, rf_model) -> pd.DataFrame:
    """Tabla comparativa de los 6 métodos para un ground truth."""
    ev = load_evaluation_frame(ground_truth=ground_truth)
    scores, y = build_method_scores(ev=ev, ground_truth=ground_truth, rf_model=rf_model)
    # Orden de filas pensado para leer FIS-actual vs FIS-optim vs Combinado de un vistazo.
    rows = [
        metrics_row("FIS actual", scores["FIS"], y),
        metrics_row("FIS optim", fis_optim.reindex(ev.index), y),
        metrics_row("Combinado", scores["Combinado"], y),
        metrics_row("EPSS", scores["EPSS"], y),
        metrics_row("RF", scores["RF"], y),
        metrics_row("CVSS", scores["CVSS"], y),
    ]
    return pd.DataFrame(rows).set_index("método")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 200)
    OPTIM_DIR.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(BEST_CONFIG.read_text(encoding="utf-8"))
    logger.info("config ganadora: a=%.3f b=%.3f c=%.3f d=%.3f high=%s",
                cfg["a"], cfg["b"], cfg["c"], cfg["d"],
                "triangular" if cfg["high_tri"] else "trapezoidal")

    fis_optim = fis_optim_priority(cfg)
    rf_model = train_rf_model()  # una vez; el RF no depende del ground truth de test

    for gt in ("kev", "multi"):
        table = evaluate(gt, fis_optim, rf_model)
        out = OPTIM_DIR / f"optim_vs_baselines_{gt}.csv"
        table.to_csv(out)
        print("\n" + "=" * 92)
        print(f"GROUND TRUTH: {gt.upper()}")
        print("=" * 92)
        print(table.to_string())
        print(f"-> {out}")

    print("\nPregunta clave (pre-req b): en MULTI, ¿FIS optim alcanza al Combinado? "
          "Comparar filas 'FIS optim' y 'Combinado' en optim_vs_baselines_multi.csv.")


if __name__ == "__main__":
    main()
