# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# Keep container logs unbuffered and avoid writing .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# This repository uses requirements.txt for dependencies.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only files needed to run the API + static demo.
COPY src /app/src
COPY web /app/web

# Ensure runtime directories exist for mounted/uploaded artifacts.
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
