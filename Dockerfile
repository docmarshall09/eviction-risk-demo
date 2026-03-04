# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV EVICTION_LAB_CSV_URL=https://eviction-lab-data-downloads.s3.amazonaws.com/data-for-analysis/county_proprietary_valid_2000_2018.csv

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src

# Build-time training: download public raw data and produce yearly artifacts.
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports
RUN python -c "from urllib.request import urlretrieve; urlretrieve('${EVICTION_LAB_CSV_URL}', '/app/data/raw/county_proprietary_valid_2000_2018.csv')"
RUN python -m src.main --task train_eviction_lab_yearly_final

# Fail fast with clear errors when expected artifacts are missing.
RUN test -f /app/models/eviction_lab_yearly_model.joblib || (echo 'ERROR: missing /app/models/eviction_lab_yearly_model.joblib after build-time training.' && exit 1)
RUN test -f /app/models/eviction_lab_yearly_model_metadata.json || (echo 'ERROR: missing /app/models/eviction_lab_yearly_model_metadata.json after build-time training.' && exit 1)
RUN test -f /app/data/processed/eviction_lab_yearly_features.csv || (echo 'ERROR: missing /app/data/processed/eviction_lab_yearly_features.csv after build-time training.' && exit 1)


FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY web /app/web

RUN mkdir -p /app/data/processed /app/models /app/reports

# Runtime only gets serving artifacts (no raw CSV, no training at startup).
COPY --from=builder /app/models/eviction_lab_yearly_model.joblib /app/models/eviction_lab_yearly_model.joblib
COPY --from=builder /app/models/eviction_lab_yearly_model_metadata.json /app/models/eviction_lab_yearly_model_metadata.json
COPY --from=builder /app/data/processed/eviction_lab_yearly_features.csv /app/data/processed/eviction_lab_yearly_features.csv

RUN groupadd --system appuser \
    && useradd --system --gid appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
