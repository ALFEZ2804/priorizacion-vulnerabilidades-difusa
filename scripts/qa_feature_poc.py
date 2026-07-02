"""QA del aporte de la feature 'PoC público' antes de integrarla en el FIS.

Mide, de forma 100% read-only y sin tocar el FIS ni los resultados oficiales, si la
presencia de un exploit público anterior al freeze (`edb_first_published` ≤
2023-12-31) aporta señal **ortogonal** a EPSS para predecir la entrada en KEV durante
2024-2025. El objetivo es decidir con datos si merece la pena integrar la feature en
el sub-FIS de amenaza (recalibrar + re-evaluar) o descartarla justificadamente.

Solo se evalúa contra el ground truth **KEV**: en el multi-source el universo excluye
los PoC pre-freeze (`dataset.load_evaluation_frame`), donde la feature sería constante.

Cuatro análisis sobre el universo KEV:
  1. Prevalencia y lift de la feature.
  2. Ortogonalidad con EPSS (positivos: PoC × bins de epss_percentile).
  3. Misses rescatables (positivos fuera del top-500 con PoC).
  4. Prueba de techo: proxy `epss_percentile + alpha·PoC` vs EPSS y FIS.

Uso:
    python scripts/qa_feature_poc.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# La consola de Windows usa cp1252 por defecto y no codifica '→', '·', etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines.epss import epss_score  # noqa: E402
from src.evaluation.dataset import KEV_FREEZE_DATE, load_evaluation_frame  # noqa: E402
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger("qa_feature_poc")

K_VALUES = (50, 100, 500)
ALPHAS = (0.25, 0.5, 1.0)
EPSS_HIGH = 0.9  # umbral "EPSS alto"; por debajo, EPSS no lo pone en cabeza
OUT_DIR = _PROJECT_ROOT / "data" / "processed" / "qa_poc"


def _metrics_row(name: str, score: pd.Series, y: pd.Series) -> dict:
    """Fila de métricas P@/R@/NDCG@K para un score, con las funciones oficiales."""
    s = np.asarray(score, dtype=float)
    yt = np.asarray(y, dtype=float)
    row: dict = {"método": name}
    for k in K_VALUES:
        row[f"P@{k}"] = round(precision_at_k(s, yt, k), 4)
        row[f"R@{k}"] = round(recall_at_k(s, yt, k), 4)
        row[f"NDCG@{k}"] = round(ndcg_at_k(s, yt, k), 4)
    return row


def _top_k_ids(score: pd.Series, k: int) -> set:
    """IDs del top-K por score descendente; los NaN van al fondo (como en métricas)."""
    s = score.copy()
    if s.isna().any():
        s = s.fillna(s.min() - 1.0)
    return set(s.sort_values(ascending=False, kind="stable").index[:k])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    # --- Universo KEV + señales -------------------------------------------------
    ev = load_evaluation_frame(ground_truth="kev")
    y = ev["y_test"].astype(bool)
    fis = ev["priority"].astype(float)
    epss = epss_score(ev)
    poc = ev["edb_first_published"].le(KEV_FREEZE_DATE).fillna(False)

    n_pos = int(y.sum())
    print("\n" + "=" * 72)
    print(f"QA FEATURE PoC | universo KEV: {len(ev)} CVEs | positivos y_test={n_pos}")
    print(f"freeze={KEV_FREEZE_DATE.date()} | feature = exploit EDB publicado <= freeze")

    # --- 1) Prevalencia y lift --------------------------------------------------
    base_rate = float(y.mean())
    p_pos_poc = float(y[poc].mean()) if poc.any() else float("nan")
    p_pos_nopoc = float(y[~poc].mean()) if (~poc).any() else float("nan")
    a1 = {
        "P(PoC) global": round(float(poc.mean()), 4),
        "n con PoC": int(poc.sum()),
        "tasa base P(y=1)": round(base_rate, 6),
        "P(PoC | y=1)": round(float(poc[y].mean()), 4),
        "P(PoC | y=0)": round(float(poc[~y].mean()), 4),
        "P(y=1 | PoC)": round(p_pos_poc, 6),
        "P(y=1 | noPoC)": round(p_pos_nopoc, 6),
        "lift = P(y=1|PoC)/base": round(p_pos_poc / base_rate, 2) if base_rate else float("nan"),
        "positivos con PoC": int(poc[y].sum()),
    }
    print("\n[1] PREVALENCIA Y LIFT")
    for k, v in a1.items():
        print(f"    {k:28s}: {v}")

    # --- 2) Ortogonalidad con EPSS (en positivos) -------------------------------
    pos = ev[y].copy()
    pos["PoC"] = poc[y]
    pos["epss_bin"] = pd.cut(
        pos["epss_percentile"],
        bins=[-0.001, 0.5, 0.7, 0.9, 1.0],
        labels=["<=0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"],
    )
    ct = pd.crosstab(pos["epss_bin"], pos["PoC"], dropna=False)
    n_pos_poc_lowepss = int(((poc[y]) & (ev.loc[y.index[y], "epss_percentile"] < EPSS_HIGH)).sum())
    print("\n[2] ORTOGONALIDAD CON EPSS (solo positivos)")
    print("    positivos por bin de epss_percentile × tiene PoC:")
    print(ct.to_string().replace("\n", "\n    "))
    print(f"    -> positivos con PoC y EPSS<{EPSS_HIGH} (los que EPSS no pone arriba): {n_pos_poc_lowepss}")

    # --- 3) Misses rescatables --------------------------------------------------
    fis_top = _top_k_ids(fis, 500)
    epss_top = _top_k_ids(epss, 500)
    pos_ids = list(ev.index[y])
    fis_miss = [c for c in pos_ids if c not in fis_top]
    epss_miss = [c for c in pos_ids if c not in epss_top]
    fis_miss_poc = [c for c in fis_miss if bool(poc[c])]
    both_miss_poc = [c for c in fis_miss if (c not in epss_top) and bool(poc[c])]
    print("\n[3] MISSES RESCATABLES (top-500)")
    print(f"    positivos fuera del top-500 FIS : {len(fis_miss)}/{n_pos}")
    print(f"    positivos fuera del top-500 EPSS: {len(epss_miss)}/{n_pos}")
    print(f"    de los miss del FIS, con PoC    : {len(fis_miss_poc)}")
    print(f"    miss de FIS *y* EPSS, con PoC   : {len(both_miss_poc)}  (ganancia ortogonal)")
    if fis_miss_poc:
        det = ev.loc[fis_miss_poc, ["epss_percentile", "priority", "edb_first_published", "kev_date_added"]]
        det = det.sort_values("epss_percentile")
        print("    detalle (positivos miss del FIS con PoC):")
        print(det.to_string().replace("\n", "\n    "))

    # --- 4) Prueba de techo: proxy EPSS + alpha·PoC -----------------------------
    rows = [_metrics_row("FIS (actual)", fis, y), _metrics_row("EPSS", epss, y)]
    poc_f = poc.astype(float)
    for alpha in ALPHAS:
        proxy = (epss + alpha * poc_f).reindex(ev.index)
        rows.append(_metrics_row(f"EPSS+{alpha}·PoC", proxy, y))
    table = pd.DataFrame(rows).set_index("método")
    print("\n[4] PRUEBA DE TECHO (proxy, sin construir el FIS)")
    print(table.to_string().replace("\n", "\n    ").replace("método", "    método", 1))

    # Conversión de R@500 a nº de CVEs para el criterio
    r500_epss = table.loc["EPSS", "R@500"]
    best_proxy = table.loc[[i for i in table.index if i.startswith("EPSS+")], "R@500"].max()
    delta_cves = round((best_proxy - r500_epss) * n_pos, 1)

    # --- Veredicto --------------------------------------------------------------
    n_pos_poc = int(poc[y].sum())
    if (delta_cves >= 2) and (len(both_miss_poc) >= 3):
        verdict = "VERDE → integrar feature (+ recalibrar/GA)"
    elif (delta_cves <= 0) or (n_pos_poc < 2):
        verdict = "ROJO → no integrar (documentar la decisión con estos números)"
    else:
        verdict = "ÁMBAR → decidir según tiempo restante (ganancia marginal)"
    print("\n" + "=" * 72)
    print("VEREDICTO")
    print(f"    positivos con PoC pre-freeze        : {n_pos_poc}/{n_pos}")
    print(f"    mejor proxy R@500                   : {best_proxy:.4f} (EPSS={r500_epss:.4f})")
    print(f"    ganancia R@500 del proxy            : {delta_cves:+} CVEs")
    print(f"    miss FIS+EPSS con PoC (ortogonales) : {len(both_miss_poc)}")
    print(f"    => {verdict}")
    print("=" * 72)

    # --- Persistencia (gitignored, no contamina resultados oficiales) -----------
    table.to_csv(OUT_DIR / "proxy_metrics.csv")
    pd.DataFrame([a1]).to_csv(OUT_DIR / "prevalence_lift.csv", index=False)
    ct.to_csv(OUT_DIR / "orthogonality_epss.csv")
    if fis_miss_poc:
        det.to_csv(OUT_DIR / "fis_miss_with_poc.csv")
    logger.info("tablas del QA guardadas en %s", OUT_DIR)


if __name__ == "__main__":
    main()
