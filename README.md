# Breast Cancer Triage Service (MLOps Assignment 02)

Educational screening-support example using sklearn's Wisconsin breast-cancer dataset. This is not a medical device. Target 0 is malignant; malignant is treated as the positive class.

## Current completion status

The source project, two DVC data versions, tracked local DVC remote, training pipeline, API, and Docker/Compose configuration are prepared. DVC training and unchanged-stage checks were run locally on the corrected 539-row dataset; logistic regression achieved 0.9756 malignant recall on the held-out test set. The public GitHub and Docker Hub repositories have been created. Docker CLI is installed, but this Codex execution environment receives “permission denied” on Docker Desktop's Linux engine named pipe, so image builds, container lifecycle, Compose runs, screenshots, and registry tags remain unverified.

Public repositories: [GitHub](https://github.com/Ayesha-Najeeb7/breast-cancer-triage-service) and [Docker Hub](https://hub.docker.com/r/ayesha078/breast-cancer-triage-api). The ignored local `.env` contains the Docker Hub username and Compose defaults only. Never put credentials in `.env` or Git.

## Prerequisites

Docker Desktop with WSL 2, Docker Compose v2, Python 3.11+, and pip. Commands below use PowerShell syntax where shell-specific.

## Restore and version data

`requirements-dev.txt` pins DVC. Create a virtual environment, install the exact local tooling, and restore the data:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-train.txt
Copy-Item .env.example .env
# edit .env and set DOCKERHUB_USERNAME

dvc pull
```

The local DVC remote is `dvc-remote/`, included in Git as required for this small classroom dataset. The tracked CSV is excluded from Git; only its `.dvc` pointer is tracked. To reproduce the original dataset, checkout the first data-version commit and run `dvc checkout`; for the corrected version, checkout the later data-version commit and run `dvc checkout`. The corrected version contains 539 rows after deleting the final 30 rows from the source export.

## Run training

```powershell
python training/prepare_data.py
# For assignment dataset version 2, restore it again after prepare_data exports all 569 rows:
dvc checkout
python training/train.py --data-path data/raw/breast_cancer.csv --output-dir artifacts --model-version 1.0.0
dvc repro train
```

Training uses seed 42, a stratified 80/20 split, and stratified 5-fold CV. It compares logistic regression and random forest Pipelines. `params.yaml` sets the training/evaluation threshold to 0.35 to meet the assignment recall gate. Runtime threshold defaults to 0.5 for a standalone image and can be set through `DECISION_THRESHOLD`; the Compose example uses 0.35 so its runtime matches the recorded evaluation threshold. Metrics and model card are emitted under `artifacts/`.

## Run the API locally

```powershell
python -m pip install -r requirements-serve.txt
$env:MODEL_PATH = 'artifacts/model.joblib'
$env:DECISION_THRESHOLD = '0.35'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. API contract: `GET /health`, `POST /predict`, `GET /model/info`, and `GET /stats`. All 30 values are nonnegative numbers; missing, invalid, and extra fields return 422. The service needs no Redis for health or predictions. `REDIS_HOST` enables a best-effort counter.

## Build and run the production image

```powershell
$env:MODEL_VERSION = '1.0.0'
$env:GIT_SHA = (git rev-parse --short HEAD)
$buildStart = Get-Date
docker build --build-arg MODEL_VERSION=$env:MODEL_VERSION --build-arg GIT_SHA=$env:GIT_SHA -t breast-cancer-triage-api:1.0.0 .
(Get-Date) - $buildStart
docker image inspect breast-cancer-triage-api:1.0.0
docker run --rm -d --name triage-api -p 8000:8000 -e DECISION_THRESHOLD=0.35 breast-cancer-triage-api:1.0.0
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model/info
docker inspect --format '{{json .State.Health}}' triage-api
docker logs triage-api
docker stop triage-api
```

For the naive baseline and rebuild timing, build `-f Dockerfile.naive -t breast-cancer-triage-api:naive .`, then change one application line, time a repeat build, and record image sizes with `docker image ls`. Revert the edit after collecting evidence.

## Run Compose

```powershell
Copy-Item .env.example .env
# edit DOCKERHUB_USERNAME and optionally HOST_PORT / DECISION_THRESHOLD
docker compose up --build -d
docker compose ps
# Send predictions to http://localhost:<HOST_PORT>/docs and confirm /stats increments
docker compose down
docker compose up -d
# check /stats; Redis named volume persists the counter
docker compose down -v
```

## Publish and verify (requires your Docker Hub account)

```powershell
docker login
docker build --build-arg MODEL_VERSION=1.0.0 --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t $env:DOCKERHUB_USERNAME/breast-cancer-triage-api:1.0.0 .
docker tag $env:DOCKERHUB_USERNAME/breast-cancer-triage-api:1.0.0 $env:DOCKERHUB_USERNAME/breast-cancer-triage-api:latest
docker push $env:DOCKERHUB_USERNAME/breast-cancer-triage-api:1.0.0
docker push $env:DOCKERHUB_USERNAME/breast-cancer-triage-api:latest
```

Create `docker-compose.hub.yml` plus `.env` in an empty directory, set `IMAGE_TAG=1.0.0`, and run `docker compose -f docker-compose.hub.yml up -d`; then verify `/health`, `/model/info`, and `/predict`. Record the registry digest from `docker image inspect` and compare it with the Docker Hub tag page.

## Evidence checklist for report

Capture terminal screenshots showing both commands and outputs for every Milestone 0 task, DVC tracking/push/restore/version switch, each training and DVC pipeline run, `/docs` prediction, naive and production builds and rebuild after code edit, container health/info/logs/inspect, Compose counter persistence and reset, Docker Hub username and tags, and clean-room pull/prediction/digest. Populate the report's three tables and replace the identity and repository URL placeholders. Do not submit screenshots that claim results that were not run.
