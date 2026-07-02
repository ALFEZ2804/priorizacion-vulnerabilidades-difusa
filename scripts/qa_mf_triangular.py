"""QA: ¿mejora la predicción si la MF 'high' de EPSS es triangular en vez de trapezoidal?

Prueba aislada del diagnóstico: la MF `high` de `epss_percentile` en
FIS-2 es `trapmf([0.7, 0.9, 1.0, 1.0])`, que satura en grado 1.0 desde 0.9 y aplasta
toda la resolución de EPSS en su zona más informativa (la cabeza del riesgo). Una MF
triangular `[0.7, 1.0, 1.0]` no satura: el grado sigue subiendo hasta 1.0, así que
preserva el orden de EPSS en [0.9, 1.0].

NO toca `src/fis/threat.py` ni los resultados oficiales: reconstruye un FIS-2 variante
en memoria, recalcula `threat` y luego `priority` (FIS-3 sin cambios) sobre todo el
universo, y mide P@/R@K contra el FIS actual y EPSS en el ground truth KEV.

Uso:
    python scripts/qa_mf_triangular.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
from src.evaluation.metrics import precision_at_k, recall_at_k  # noqa: E402

K_VALUES = (50, 100, 500)
SCORED = _PROJECT_ROOT / "data/processed/scored.parquet"
TEST = _PROJECT_ROOT / "data/processed/test.parquet"
OUT_DIR = _PROJECT_ROOT / "data/processed/qa_mf"

# Variantes de la MF 'high' de epss_percentile. 4 puntos = trapmf, 3 = trimf.
HIGH_VARIANTS = {
    "trapezoidal (actual)": [0.7, 0.9, 1.0, 1.0],
    "triangular [0.7,1,1]": [0.7, 1.0, 1.0],
    "triangular [0.9,1,1]": [0.9, 1.0, 1.0],
}


def build_epss_antecedent(high_points: list[float]) -> ctrl.Antecedent:
    """Antecedente epss_percentile con low/medium fijos y 'high' parametrizable."""
    var = ctrl.Antecedent(universe(0.0, 1.0, 0.01), "epss_percentile")
    var["low"] = fuzz.trapmf(var.universe, [0.0, 0.0, 0.3, 0.5])
    var["medium"] = fuzz.trapmf(var.universe, [0.3, 0.5, 0.7, 0.9])
    if len(high_points) == 3:
        var["high"] = fuzz.trimf(var.universe, high_points)
    else:
        var["high"] = fuzz.trapmf(var.universe, high_points)
    return var


def build_threat_variant(high_points: list[float]) -> FuzzyInference:
    """FIS-2 con la MF 'high' indicada; delta, consecuente y reglas sin cambios."""
    pct = build_epss_antecedent(high_points)
    dlt = build_epss_delta_antecedent()
    thr = build_threat_consequent()
    return FuzzyInference([pct, dlt], thr, _build_threat_rules(pct, dlt, thr))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(SCORED).set_index("cve_id")
    test = pd.read_parquet(TEST).set_index("cve_id")

    # Universo KEV (no in_kev al freeze) y target.
    ev_mask = ~test["in_kev"].astype(bool)
    y = test.loc[ev_mask, "y_test"].astype(bool)
    epss = test.loc[ev_mask, "epss_percentile"].astype(float)

    fis3 = build_priority_fis()  # FIS-3 sin cambios, reutilizado en todas las variantes
    sev = scored["severity"].to_numpy(dtype=float)
    logcpe = scored["log_cpe"].to_numpy(dtype=float)
    pct_in = scored["epss_percentile"].to_numpy(dtype=float)
    dlt_in = scored["epss_delta"].to_numpy(dtype=float)
    in_kev = scored["in_kev"].astype(int).to_numpy()

    rows = []
    for name, high in HIGH_VARIANTS.items():
        print(f"... calculando variante: {name}")
        thr = build_threat_variant(high).infer_batch(epss_percentile=pct_in, epss_delta=dlt_in)
        praw = fis3.infer_batch(severity=sev, threat=thr, log_cpe=logcpe)
        prio = pd.Series(praw + in_kev * 2.0, index=scored.index).clip(0.0, 10.0)
        score = prio.reindex(y.index)
        row = {"variante FIS": name}
        for k in K_VALUES:
            row[f"P@{k}"] = round(precision_at_k(score.values, y.values, k), 3)
            row[f"R@{k}"] = round(recall_at_k(score.values, y.values, k), 3)
        rows.append(row)

        # ¿se recupera la monotonía de threat en la cabeza?
        thr_s = pd.Series(thr, index=scored.index)
        tramos = [(0.90, 0.95), (0.95, 0.99), (0.99, 1.0)]
        medias = [
            thr_s[(scored.epss_percentile >= lo) & (scored.epss_percentile <= hi)].mean()
            for lo, hi in tramos
        ]
        print(f"    threat medio en epss [0.90-0.95 / 0.95-0.99 / 0.99-1.0]: "
              f"{medias[0]:.3f} / {medias[1]:.3f} / {medias[2]:.3f}")

    # Referencia EPSS
    ref = {"variante FIS": "-- EPSS (ref) --"}
    for k in K_VALUES:
        ref[f"P@{k}"] = round(precision_at_k(epss.values, y.values, k), 3)
        ref[f"R@{k}"] = round(recall_at_k(epss.values, y.values, k), 3)
    rows.append(ref)

    table = pd.DataFrame(rows).set_index("variante FIS")
    print("\n" + "=" * 72)
    print(f"MF 'high' de EPSS: trapezoidal vs triangular | universo KEV, {int(y.sum())} positivos")
    print("=" * 72)
    print(table.to_string())
    table.to_csv(OUT_DIR / "mf_triangular_metrics.csv")


if __name__ == "__main__":
    main()
