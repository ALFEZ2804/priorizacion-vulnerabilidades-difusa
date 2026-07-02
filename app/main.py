"""Aplicación web del sistema de priorización difuso.

Expone el ranking de vulnerabilidades puntuado por el FIS y permite explorarlo:
filtrar y ordenar la tabla, abrir el detalle de un CVE con su **explicación regla a
regla** (XAI, vía `src.fis.explain`) y comparar su posición entre los cinco métodos.
El ranking se carga una sola vez al arrancar desde `scored.parquet`; el frontend
estático se sirve desde `app/static` y `app/templates`.

Arranque local:
    uvicorn app.main:app --reload
Documentación interactiva en /docs (la genera FastAPI).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.baselines.combined import combined_score
from src.baselines.cvss import cvss_score
from src.baselines.epss import epss_score
from src.baselines.random_forest import load_model, rf_score
from src.fis.explain import explain_cve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
APP_DIR = Path(__file__).resolve().parent

# Métodos comparados y columnas que el frontend muestra en la tabla.
METHODS = ["FIS", "CVSS", "EPSS", "Combinado", "RF"]
_SORTABLE = {
    "priority": "priority", "cvss": "cvss_v3_score", "epss": "epss_percentile",
    "severity": "severity", "threat": "threat",
}

STATE: dict = {}


def _load_state() -> dict:
    """Carga el ranking y precomputa scores y rangos de los cinco métodos."""
    scored = pd.read_parquet(DATA / "scored.parquet").set_index("cve_id")

    scores = {
        "FIS": scored["priority"],
        "CVSS": cvss_score(scored),
        "EPSS": epss_score(scored),
        "Combinado": combined_score(scored),
    }
    try:  # el RF necesita el modelo persistido; si falta, se omite con gracia
        scores["RF"] = rf_score(load_model(), scored)
    except FileNotFoundError:
        scores["RF"] = pd.Series(float("nan"), index=scored.index)

    # Rango de cada método (1 = más prioritario); los NaN van al fondo.
    ranks = {
        m: s.rank(ascending=False, method="min", na_option="bottom").astype(int)
        for m, s in scores.items()
    }

    results = None
    if (DATA / "results.csv").exists():
        results = pd.read_csv(DATA / "results.csv", index_col=0)
    robustness = None
    if (DATA / "robustness.csv").exists():
        robustness = pd.read_csv(DATA / "robustness.csv")

    return {"scored": scored, "scores": scores, "ranks": ranks,
            "results": results, "robustness": robustness}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.update(_load_state())
    yield
    STATE.clear()


app = FastAPI(
    title="Priorización difusa de vulnerabilidades",
    description="Explorador del ranking del sistema de inferencia difuso jerárquico, "
                "con explicación regla a regla y comparación entre métodos.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Sirve la página principal del explorador."""
    return FileResponse(APP_DIR / "templates" / "index.html")


@app.get("/api/stats")
def stats() -> dict:
    """Estadísticas globales y datos agregados para las gráficas."""
    scored = STATE["scored"]
    total = int(len(scored))
    n_kev = int(scored["in_kev"].sum())
    # Histograma de la prioridad (20 cubos) para la gráfica de distribución.
    counts, edges = pd.cut(scored["priority"].dropna(), bins=20, retbins=True)
    hist = counts.value_counts(sort=False)
    distribution = {
        "edges": [round(float(e), 2) for e in edges],
        "counts": [int(c) for c in hist.tolist()],
    }
    results = STATE.get("results")
    methods_table = (
        results.reset_index().rename(columns={"index": "method"}).to_dict("records")
        if results is not None else []
    )
    robustness = STATE.get("robustness")
    robustness_rows = robustness.to_dict("records") if robustness is not None else []
    return {
        "total": total,
        "n_kev": n_kev,
        "kev_rate": round(n_kev / total, 4),
        "distribution": distribution,
        "methods_table": methods_table,
        "robustness": robustness_rows,
    }


@app.get("/api/vulnerabilities")
def vulnerabilities(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort: str = Query("priority"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    search: str | None = Query(None, description="Subcadena del CVE-ID"),
    only_kev: bool = Query(False),
) -> dict:
    """Página del ranking de vulnerabilidades, ordenable y filtrable."""
    if sort not in _SORTABLE:
        raise HTTPException(400, f"sort debe ser uno de {list(_SORTABLE)}")
    scored = STATE["scored"]
    ranks_fis = STATE["ranks"]["FIS"]

    df = scored
    if only_kev:
        df = df[df["in_kev"]]
    if search:
        df = df[df.index.str.contains(search.strip().upper(), na=False)]

    df = df.sort_values(_SORTABLE[sort], ascending=(order == "asc"), na_position="last")
    total = int(len(df))
    page = df.iloc[offset:offset + limit]

    rows = [
        {
            "cve_id": cve_id,
            "rank_fis": int(ranks_fis.loc[cve_id]),
            "priority": _num(r["priority"]),
            "severity": _num(r["severity"]),
            "threat": _num(r["threat"]),
            "cvss": _num(r["cvss_v3_score"]),
            "epss": _num(r["epss_percentile"]),
            "in_kev": bool(r["in_kev"]),
        }
        for cve_id, r in page.iterrows()
    ]
    return {"total": total, "offset": offset, "limit": limit, "rows": rows}


@app.get("/api/vulnerability/{cve_id}")
def vulnerability(cve_id: str) -> dict:
    """Detalle de un CVE: entradas, explicación regla a regla y comparación."""
    scored = STATE["scored"]
    cve_id = cve_id.strip().upper()
    if cve_id not in scored.index:
        raise HTTPException(404, f"{cve_id} no está en el conjunto puntuado")
    r = scored.loc[cve_id]

    explanation = explain_cve(
        exploitability_subscore=float(r["exploitability_subscore"]),
        impact_subscore=float(r["impact_subscore"]),
        epss_percentile=float(r["epss_percentile"]) if pd.notna(r["epss_percentile"]) else 0.0,
        epss_delta=float(r["epss_delta"]) if pd.notna(r["epss_delta"]) else 0.0,
        log_cpe=float(r["log_cpe"]),
        in_kev=bool(r["in_kev"]),
    )
    comparison = [
        {
            "method": m,
            "score": _num(STATE["scores"][m].loc[cve_id]),
            "rank": int(STATE["ranks"][m].loc[cve_id]),
        }
        for m in METHODS
    ]
    return {
        "cve_id": cve_id,
        "published": _date(r.get("published")),
        "in_kev": bool(r["in_kev"]),
        "kev_date_added": _date(r.get("kev_date_added")),
        "inputs": {
            "cvss_v3_score": _num(r["cvss_v3_score"]),
            "exploitability_subscore": _num(r["exploitability_subscore"]),
            "impact_subscore": _num(r["impact_subscore"]),
            "epss_percentile": _num(r["epss_percentile"]),
            "epss_delta": _num(r["epss_delta"]),
            "log_cpe": _num(r["log_cpe"]),
        },
        "explanation": explanation,
        "comparison": comparison,
    }


def _num(value) -> float | None:
    """Float redondeado o None si es NaN (para JSON válido)."""
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _date(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())
