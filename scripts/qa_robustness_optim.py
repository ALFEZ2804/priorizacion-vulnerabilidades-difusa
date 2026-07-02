"""QA: ¿la config optimizada del FIS-2 conserva la robustez del FIS frente a EPSS
faltante/ruidoso?.

El optimizador (`scripts/optimize_fis.py`) eligió una MF `high` de `epss_percentile`
**triangular** y más selectiva (cortes `data/processed/optim/best_config.json`), que mejora
la cabeza del ranking en test (KEV R@500 0,108 → 0,162). Pero la robustez es el argumento
estrella del TFG: el FIS degrada con más gracia que el RF cuando EPSS falta o llega con ruido.
Una MF triangular que **no satura** depende
más finamente del valor exacto de EPSS, así que **podría volverse más frágil justo ahí**. Este
script lo verifica replicando los 3 escenarios de robustez con TRES curvas:
FIS-actual (trapezoidal) · FIS-optimizado (triangular) · RF.

NO toca producción: reconstruye los dos FIS-2 en memoria (vía `build_threat_variant`, idéntica
a la de `optimize_fis.py`) y escribe solo en `data/processed/optim/`.

Uso:
    python scripts/qa_robustness_optim.py
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # sin display (se ejecuta headless)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skfuzzy as fuzz  # noqa: E402
from skfuzzy import control as ctrl  # noqa: E402

from src.evaluation.dataset import load_evaluation_frame  # noqa: E402
from src.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.baselines.random_forest import load_model, rf_score  # noqa: E402
from src.fis import FuzzyInference  # noqa: E402
from src.fis._universe import universe  # noqa: E402
from src.fis.priority import build_priority_fis  # noqa: E402
from src.fis.severity import compute_severity  # noqa: E402
from src.fis.threat import (  # noqa: E402
    _build_threat_rules,
    build_epss_delta_antecedent,
    build_threat_consequent,
)

logger = logging.getLogger("qa_robustness_optim")

OUT_DIR = _PROJECT_ROOT / "data/processed/optim"
BEST_CONFIG = OUT_DIR / "best_config.json"

# Config vigente en src/fis/threat.py: build_threat_variant la reproduce exactamente
# (low=[0,0,.3,.5], medium=[.3,.5,.7,.9], high trapmf=[.7,.9,1,1]). Verificado por el
# sanity check de optimize_fis.py (R@500=0,108 == results.csv).
ACTUAL_CFG = {"a": 0.3, "b": 0.5, "c": 0.7, "d": 0.9, "high_tri": False}

SA = 5  # semillas para escenarios de datos faltantes
SC = 3  # semillas para ruido (recompute completo, más caro)
MISSING_FRACS = (0.1, 0.3, 0.5)
NOISE_SIGMAS = (0.05, 0.1, 0.2)

# Referencias de baseline para el sanity check (KEV, sin degradar).
REF_R500_ACTUAL = 0.108  # results.csv
REF_R500_OPTIM = 0.162   # data/processed/optim/best_vs_baselines.csv


def build_threat_variant(a: float, b: float, c: float, d: float, high_tri: bool) -> FuzzyInference:
    """FIS-2 con los cortes (a<b<c<d) de epss_percentile; delta/reglas sin cambios.

    Idéntica a la de `scripts/optimize_fis.py` para que ambos scripts evalúen el mismo
    objeto. `high_tri=False` reproduce la MF trapezoidal de producción.
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    t0 = time.perf_counter()

    # --- Universo, ground truth, RF, FIS-3 (compartidos) -----------------------
    ev = load_evaluation_frame(ground_truth="kev")
    y = ev["y_test"].astype(int).to_numpy()
    rf = load_model()
    fis3 = build_priority_fis()

    # Dos FIS-2: el actual (trapezoidal) y el optimizado (triangular).
    cfg = json.loads(BEST_CONFIG.read_text(encoding="utf-8"))
    thr_actual = build_threat_variant(**ACTUAL_CFG)
    thr_optim = build_threat_variant(cfg["a"], cfg["b"], cfg["c"], cfg["d"], cfg["high_tri"])
    logger.info("config optimizada: a=%.3f b=%.3f c=%.3f d=%.3f high=%s",
                cfg["a"], cfg["b"], cfg["c"], cfg["d"],
                "triangular" if cfg["high_tri"] else "trapezoidal")

    # severity NO depende de EPSS: se calcula una vez y se reutiliza en A y C (escenarios
    # que solo degradan EPSS). En B (subscores faltantes) se recomputa sobre el subset.
    sev_clean = compute_severity(ev)["severity"]
    log_cpe = ev["log_cpe"].to_numpy(dtype=float)
    in_kev = ev["in_kev"].astype(int).to_numpy()

    def fis_priority(df: pd.DataFrame, threat_fis: FuzzyInference, severity: pd.Series) -> pd.Series:
        """priority del FIS = FIS-2(variante) -> FIS-3 -> +2*in_kev clip[0,10]."""
        threat = threat_fis.infer_batch(
            epss_percentile=df["epss_percentile"].to_numpy(dtype=float),
            epss_delta=df["epss_delta"].to_numpy(dtype=float),
        )
        praw = fis3.infer_batch(
            severity=severity.to_numpy(dtype=float),
            threat=threat,
            log_cpe=df["log_cpe"].to_numpy(dtype=float),
        )
        boost = df["in_kev"].astype(int).to_numpy() * 2.0
        return pd.Series(np.clip(praw + boost, 0.0, 10.0), index=df.index)

    # --- Baselines sin degradar + sanity check ---------------------------------
    base_actual = fis_priority(ev, thr_actual, sev_clean)
    base_optim = fis_priority(ev, thr_optim, sev_clean)
    rf_base = rf_score(rf, ev)

    r500_actual = recall_at_k(base_actual.to_numpy(), y, 500)
    r500_optim = recall_at_k(base_optim.to_numpy(), y, 500)
    print(f"\n[SANITY] universo={len(ev)} positivos KEV={int(y.sum())}")
    print(f"  FIS-actual R@500 = {r500_actual:.4f} (esperado ~{REF_R500_ACTUAL})")
    print(f"  FIS-optim  R@500 = {r500_optim:.4f} (esperado ~{REF_R500_OPTIM})")
    print(f"  RF         R@500 = {recall_at_k(rf_base.to_numpy(), y, 500):.4f}")
    if abs(r500_actual - REF_R500_ACTUAL) > 0.01 or abs(r500_optim - REF_R500_OPTIM) > 0.01:
        logger.warning("¡el sanity check NO reproduce los baselines! revisar antes de fiarse.")

    # --- Helpers de degradación (mismas filas para los 3 métodos) --------------
    _CVSS_COLS = {"exploitability_subscore", "impact_subscore"}

    def fis_missing(mask, vals: dict, threat_fis: FuzzyInference, base: pd.Series) -> pd.Series:
        """Recomputa priority solo en las filas degradadas (exacto y rápido)."""
        sub = ev.loc[mask].copy()
        for c, v in vals.items():
            sub[c] = v
        sev_sub = (compute_severity(sub)["severity"] if _CVSS_COLS & vals.keys()
                   else sev_clean.loc[mask])
        out = base.copy()
        out.loc[mask] = fis_priority(sub, threat_fis, sev_sub)
        return out

    def rf_missing(mask, cols: list[str]) -> pd.Series:
        df = ev.copy()
        df.loc[mask, cols] = np.nan  # el imputer mediana del pipeline lo maneja
        return rf_score(rf, df)

    def fis_noise(noise: dict, threat_fis: FuzzyInference) -> pd.Series:
        df = ev.copy()
        for c, v in noise.items():
            df[c] = v
        return fis_priority(df, threat_fis, sev_clean)  # severity intacta

    def rf_noise(noise: dict) -> pd.Series:
        df = ev.copy()
        for c, v in noise.items():
            df[c] = v
        return rf_score(rf, df)

    def row(escenario: str, nivel: float, metodo: str, scores: pd.Series, seed: int = -1) -> dict:
        a = scores.to_numpy()
        return {"escenario": escenario, "nivel": nivel, "metodo": metodo, "seed": seed,
                "R@100": recall_at_k(a, y, 100), "R@500": recall_at_k(a, y, 500),
                "NDCG@100": ndcg_at_k(a, y, 100)}

    records = [row("baseline", 0.0, "FIS_actual", base_actual),
               row("baseline", 0.0, "FIS_optim", base_optim),
               row("baseline", 0.0, "RF", rf_base)]

    # A) EPSS faltante (imputación neutra a la MF media) -------------------------
    epss_neutral = {"epss_percentile": 0.5, "epss_delta": 0.0}
    for X in MISSING_FRACS:
        for seed in range(SA):
            m = ev.sample(frac=X, random_state=seed).index
            records.append(row("A_epss_missing", X, "FIS_actual", fis_missing(m, epss_neutral, thr_actual, base_actual), seed))
            records.append(row("A_epss_missing", X, "FIS_optim", fis_missing(m, epss_neutral, thr_optim, base_optim), seed))
            records.append(row("A_epss_missing", X, "RF", rf_missing(m, ["epss_percentile", "epss_delta"]), seed))
        logger.info("A_epss_missing %.0f%% hecho | %.0fs", X * 100, time.perf_counter() - t0)

    # B) Subscores CVSS faltantes (imputación neutra al punto medio) -------------
    cvss_neutral = {"exploitability_subscore": 2.0, "impact_subscore": 3.0}
    for X in MISSING_FRACS:
        for seed in range(SA):
            m = ev.sample(frac=X, random_state=100 + seed).index
            records.append(row("B_cvss_missing", X, "FIS_actual", fis_missing(m, cvss_neutral, thr_actual, base_actual), seed))
            records.append(row("B_cvss_missing", X, "FIS_optim", fis_missing(m, cvss_neutral, thr_optim, base_optim), seed))
            records.append(row("B_cvss_missing", X, "RF", rf_missing(m, ["exploitability_subscore", "impact_subscore"]), seed))
        logger.info("B_cvss_missing %.0f%% hecho | %.0fs", X * 100, time.perf_counter() - t0)

    # C) Ruido gaussiano en EPSS ------------------------------------------------
    for sg in NOISE_SIGMAS:
        for seed in range(SC):
            r = np.random.default_rng(200 + seed)
            ep = np.clip(ev["epss_percentile"].to_numpy() + r.normal(0, sg, len(ev)), 0, 1)
            ed = np.clip(ev["epss_delta"].to_numpy() + r.normal(0, sg, len(ev)), -1, 1)
            nz = {"epss_percentile": ep, "epss_delta": ed}
            records.append(row("C_epss_noise", sg, "FIS_actual", fis_noise(nz, thr_actual), seed))
            records.append(row("C_epss_noise", sg, "FIS_optim", fis_noise(nz, thr_optim), seed))
            records.append(row("C_epss_noise", sg, "RF", rf_noise(nz), seed))
        logger.info("C_epss_noise sigma=%.2f hecho | %.0fs", sg, time.perf_counter() - t0)

    # --- Agregación y volcado --------------------------------------------------
    res = pd.DataFrame(records)
    agg = res.groupby(["escenario", "nivel", "metodo"], as_index=False)[["R@100", "R@500", "NDCG@100"]].mean()
    agg.to_csv(OUT_DIR / "robustness_optim.csv", index=False)

    print("\n" + "=" * 72)
    print("ROBUSTEZ: FIS-actual vs FIS-optimizado vs RF (ground truth KEV)")
    print("=" * 72)
    for scn in ("A_epss_missing", "B_cvss_missing", "C_epss_noise"):
        print(f"\n--- {scn} ---")
        piv = agg[agg.escenario == scn].pivot(index="nivel", columns="metodo", values=["R@100", "R@500"])
        print(piv.round(4).to_string())

    # --- Figuras (3 curvas por escenario) --------------------------------------
    colors = {"FIS_actual": "#1f77b4", "FIS_optim": "#2ca02c", "RF": "#d62728"}
    base = agg[agg.escenario == "baseline"]

    def plot_scn(scn: str, xlabel: str, fname: str, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ["R@100", "R@500"]):
            for method, color in colors.items():
                d = agg[(agg.escenario == scn) & (agg.metodo == method)].sort_values("nivel")
                b = base[base.metodo == method][metric].iloc[0]
                xs = [0.0] + d.nivel.tolist()
                ys = [b] + d[metric].tolist()
                ax.plot(xs, ys, "o-", label=method, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.set_title(f"{title}: {metric}")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)

    plot_scn("A_epss_missing", "fracción EPSS faltante", "robustez_optim_epss_missing.png", "EPSS faltante")
    plot_scn("B_cvss_missing", "fracción subscores CVSS faltantes", "robustez_optim_cvss_missing.png", "Subscores CVSS faltantes")
    plot_scn("C_epss_noise", "sigma ruido EPSS", "robustez_optim_epss_noise.png", "Ruido en EPSS")

    # --- Análisis: punto de cruce y % degradación vs baseline propio -----------
    print("\n" + "=" * 72)
    print("PUNTO DE CRUCE (primer nivel donde FIS >= RF) y % degradación R@500")
    print("=" * 72)
    for scn in ("A_epss_missing", "B_cvss_missing", "C_epss_noise"):
        print(f"\n{scn}:")
        for fis in ("FIS_actual", "FIS_optim"):
            b = base[base.metodo == fis]["R@500"].iloc[0]
            cross = {"R@100": None, "R@500": None}
            for lvl in sorted(agg[agg.escenario == scn].nivel.unique()):
                for metric in ("R@100", "R@500"):
                    f = agg[(agg.escenario == scn) & (agg.nivel == lvl) & (agg.metodo == fis)][metric].iloc[0]
                    rr = agg[(agg.escenario == scn) & (agg.nivel == lvl) & (agg.metodo == "RF")][metric].iloc[0]
                    if f >= rr and cross[metric] is None:
                        cross[metric] = lvl
            worst = agg[(agg.escenario == scn) & (agg.metodo == fis)].sort_values("nivel")["R@500"].iloc[-1]
            deg = (worst - b) / b * 100 if b else float("nan")
            print(f"  {fis}: adelanta a RF en R@100>={cross['R@100']} R@500>={cross['R@500']} | "
                  f"R@500 baseline={b:.4f} -> nivel máx={worst:.4f} ({deg:+.0f}%)")

    print(f"\ntiempo total: {time.perf_counter() - t0:.0f}s | resultados en {OUT_DIR}")


if __name__ == "__main__":
    main()
