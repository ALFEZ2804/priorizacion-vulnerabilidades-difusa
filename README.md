# Priorización de vulnerabilidades con lógica difusa

Trabajo de Fin de Grado. Un sistema de inferencia difusa que ordena vulnerabilidades por
prioridad cruzando tres fuentes públicas: la severidad técnica de CVSS (NVD), la
probabilidad de explotación que estima EPSS y las vulnerabilidades con explotación ya
confirmada del catálogo CISA KEV. En lugar de un número opaco, devuelve una prioridad
continua y puede explicar, regla a regla, por qué cada CVE acabó donde acabó.

Para ver si merece la pena, el ranking se compara con CVSS a secas, con EPSS, con una
combinación de ambos y con un Random Forest. La validación es hacia atrás: de las
vulnerabilidades que puso arriba, cuántas entraron de verdad en KEV después.

La idea de fondo es sencilla. Priorizar solo con CVSS se queda corto, porque mide lo
grave que es un fallo, no lo probable que es que lo exploten. Los modelos de datos
aciertan más, pero son cajas negras. Esto busca el punto medio: competir en precisión
sin perder la capacidad de explicar cada decisión, y aguantar cuando los datos llegan
incompletos o con ruido.

## Qué hay dentro

- `src/fis/`: el sistema difuso jerárquico. Tres subsistemas encadenados (severidad,
  amenaza y prioridad) con la explicación regla a regla. Es el núcleo del trabajo.
- `src/pipeline/`: ingesta y unificación de NVD, EPSS y KEV, con un corte temporal que
  evita que la evaluación haga trampa.
- `src/baselines/`: los métodos contra los que se compara (CVSS, EPSS, combinado y
  Random Forest).
- `src/evaluation/`: métricas de ranking (Precision@K, Recall@K, NDCG), significancia por
  bootstrap y análisis de robustez.
- `app/`: una web en FastAPI para explorar el ranking y ver la explicación de cada CVE.

## Estructura

```
src/
  pipeline/     ingesta y unificación de las fuentes
  fis/          sistema de inferencia difusa jerárquico
  baselines/    CVSS, EPSS, combinado, Random Forest
  evaluation/   métricas, comparativa, significancia
app/            aplicación web (FastAPI + frontend estático)
scripts/        orquestadores del pipeline, evaluación y calibración
tests/          pruebas unitarias (pytest)
data/           dataset unificado y resultados
```

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Para ejecutarlo no hace falta ninguna clave: los datos ya procesados van incluidos en
`data/processed/`. La `NVD_API_KEY` (gratuita, en
<https://nvd.nist.gov/developers/request-an-api-key>) solo se necesita si quieres
reconstruir el dataset desde las APIs; en ese caso, copia `.env.example` a `.env` y ponla
ahí. Sin clave, NVD también responde, solo que más despacio.

## Con Docker (sin instalar nada)

Si tienes Docker es lo más cómodo, porque no necesitas ni entorno ni dependencias:

```bash
docker build -t fuzzy-vuln .
docker run -p 8000:8000 fuzzy-vuln    # abre http://localhost:8000
```

Los datos viajan dentro de la imagen, así que la app levanta sin descargar nada. Y si
quieres correr la evaluación o los tests dentro del contenedor:

```bash
docker run --rm fuzzy-vuln python scripts/run_evaluation.py
docker run --rm fuzzy-vuln pytest
```

## Uso

Como los datos procesados vienen incluidos, puedes ir directo a la evaluación o a la app.
Los dos primeros pasos solo hacen falta si quieres reconstruir el dataset de cero.

```bash
# (opcional) 1. Reconstruir el dataset unificado desde las APIs
python scripts/build_dataset.py

# (opcional) 2. Puntuar el dataset con el sistema difuso jerárquico
python scripts/score_dataset.py

# 3. Evaluar el FIS frente a los métodos de referencia
python scripts/run_evaluation.py

# 4. Significancia estadística de la comparativa
python scripts/run_significance.py

# 5. Levantar la aplicación web
uvicorn app.main:app --reload   # http://127.0.0.1:8000  (docs en /docs)
```

## Reproducibilidad

En `data/processed/` está todo lo necesario: el dataset unificado, los splits temporales,
el ranking ya puntuado, el modelo Random Forest entrenado y los CSV de resultados. Con
eso, la evaluación y la web corren sin reconstruir nada ni pedir clave. Lo único que no
se versiona es `data/raw/`, los JSON crudos de las APIs (varios cientos de MB), que se
regeneran con `scripts/build_dataset.py`.

## Pruebas

```bash
pytest
```

## Licencia

Trabajo de Fin de Grado. Uso académico.
