"""Tests del backend de la web.

Usan `TestClient`, que dispara el `lifespan` (carga de `scored.parquet` y del
modelo RF) sin levantar un servidor real. Se saltan si los artefactos de datos no
están presentes (clones limpios / CI), igual que los smoke del pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DATA = Path("data/processed")
pytestmark = pytest.mark.skipif(
    not (_DATA / "scored.parquet").exists() or not (_DATA / "rf_model.joblib").exists(),
    reason="requiere scored.parquet y rf_model.joblib generados",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:  # el 'with' activa el lifespan (carga el estado)
        yield c


def test_index_sirve_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Priorización difusa" in r.text


def test_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 92350
    assert body["n_kev"] > 0
    assert {row["method"] for row in body["methods_table"]} >= {"FIS", "CVSS", "EPSS"}
    assert len(body["distribution"]["counts"]) == 20


def test_ranking_ordenado_por_prioridad(client):
    r = client.get("/api/vulnerabilities?limit=20&sort=priority&order=desc")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    prioridades = [row["priority"] for row in body["rows"]]
    assert prioridades == sorted(prioridades, reverse=True)


def test_filtro_solo_kev(client):
    r = client.get("/api/vulnerabilities?only_kev=true&limit=10")
    assert all(row["in_kev"] for row in r.json()["rows"])


def test_sort_invalido_da_400(client):
    assert client.get("/api/vulnerabilities?sort=inexistente").status_code == 400


def test_detalle_con_explicacion_y_comparacion(client):
    r = client.get("/api/vulnerability/CVE-2020-17519")
    assert r.status_code == 200
    body = r.json()
    # La prioridad explicada coincide con la del ranking (mismo motor).
    assert body["explanation"]["priority"] == pytest.approx(8.92, abs=0.05)
    assert body["explanation"]["summary"]
    metodos = {c["method"] for c in body["comparison"]}
    assert metodos == {"FIS", "CVSS", "EPSS", "Combinado", "RF"}
    # El FIS lo coloca en la cabeza y CVSS muy por detrás (caso de estudio).
    rank = {c["method"]: c["rank"] for c in body["comparison"]}
    assert rank["FIS"] < rank["CVSS"]


def test_cve_inexistente_da_404(client):
    assert client.get("/api/vulnerability/CVE-0000-00000").status_code == 404
