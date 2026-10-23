FROM python:3.12.11-slim-bookworm AS trainer
ARG MODEL_VERSION=1.0.0
WORKDIR /build
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY training ./training
COPY params.yaml ./params.yaml
COPY data/raw/breast_cancer.csv ./data/raw/breast_cancer.csv
RUN python training/train.py --data-path data/raw/breast_cancer.csv --output-dir /model --model-version ${MODEL_VERSION:-1.0.0}

FROM python:3.12.11-slim-bookworm AS runtime
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="Breast cancer triage API" org.opencontainers.image.version="$MODEL_VERSION" org.opencontainers.image.revision="$GIT_SHA"
ENV MODEL_PATH=/app/artifacts/model.joblib DECISION_THRESHOLD=0.5 LOG_LEVEL=INFO REDIS_PORT=6379 MODEL_VERSION=$MODEL_VERSION GIT_SHA=$GIT_SHA PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && useradd --create-home --uid 10001 appuser
COPY --from=trainer /model/ ./artifacts/
COPY app ./app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD ["python","-c","import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
ENTRYPOINT ["uvicorn"]
CMD ["app.main:app","--host","0.0.0.0","--port","8000"]


