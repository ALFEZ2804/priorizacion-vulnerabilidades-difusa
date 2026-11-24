# Explainable Fuzzy Vulnerability Prioritization

[![Tests](https://github.com/ALFEZ2804/priorizacion-vulnerabilidades-difusa/actions/workflows/tests.yml/badge.svg)](https://github.com/ALFEZ2804/priorizacion-vulnerabilidades-difusa/actions/workflows/tests.yml)

Bachelor's thesis project. A hierarchical fuzzy-inference system that prioritizes vulnerabilities by combining three public sources: CVSS severity from NVD, EPSS exploitation probability, and confirmed exploitation in the CISA KEV catalog. Instead of an opaque score, it produces a continuous priority and can explain which rules contributed to each result.

The system is compared against CVSS alone, EPSS, a combined baseline, and a Random Forest. The evaluation is temporal: it measures how many vulnerabilities ranked highly were later added to KEV.

CVSS measures how severe a vulnerability is, but not how likely it is to be exploited. Machine-learning models can improve prediction but are harder to explain. This project explores a middle ground: competitive ranking quality with traceable decisions and robustness to missing or noisy inputs.

## Repository map

- src/fis/: hierarchical fuzzy inference system for severity, threat, and priority, including rule-level explanations.
- src/pipeline/: ingestion and unification of NVD, EPSS, and KEV data, with temporal cutoffs to avoid evaluation leakage.
- src/baselines/: CVSS, EPSS, combined, and Random Forest baselines.
- src/evaluation/: ranking metrics, bootstrap significance tests, and robustness analyses.
- app/: FastAPI web application for exploring rankings and explanations.
- scripts/: dataset construction, evaluation, and calibration.
- tests/: pytest suite.
- data/processed/: the processed dataset, model, and evaluation outputs used by the app.

## Requirements

- Python 3.11 or newer
- Docker, or Python and pip for a local environment

No API key is needed to run the web app or the evaluation on the committed processed data. NVD_API_KEY is optional and only needed to rebuild the dataset from the source APIs. Without it, NVD requests may be slower.

## Try it with Docker

The repository includes a Dockerfile with the processed data and runtime dependencies:

~~~sh
docker build -t fuzzy-vulnerability-prioritization .
docker run --rm -p 8000:8000 fuzzy-vulnerability-prioritization
~~~

Open http://localhost:8000.

To run the evaluation or tests in the same image:

~~~sh
docker run --rm fuzzy-vulnerability-prioritization python scripts/run_evaluation.py
docker run --rm fuzzy-vulnerability-prioritization pytest
~~~

## Run without Docker

Create a virtual environment and install dependencies:

~~~sh
python -m venv .venv
~~~

On Windows:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

On macOS or Linux:

~~~sh
source .venv/bin/activate
~~~

Then install and start the app:

~~~sh
pip install -r requirements.txt
uvicorn app.main:app --reload
~~~

Open http://127.0.0.1:8000. The interactive API documentation is available at /docs.

## Rebuild and evaluate the dataset

The processed dataset and evaluation outputs are included, so rebuilding is optional. To reproduce the pipeline from the source APIs:

~~~sh
python scripts/build_dataset.py
python scripts/score_dataset.py
python scripts/run_evaluation.py
python scripts/run_significance.py
~~~

The NVD API key can be configured in .env based on .env.example. Raw API data is not committed; the build script downloads it as needed.

## Tests

~~~sh
pytest
~~~

## Docker notes

The image includes the processed dataset, allowing the web application and evaluation to run without fetching source data. Docker is optional; the same commands can be run in a local Python environment.

