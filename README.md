# cosign-public-data-poc

This repository is a public-data underwriting risk proof of concept in Python.
It will pull public datasets and produce a basic risk score.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

## Next steps

Next, we will add public data ingestion pipelines and build a baseline risk model.
