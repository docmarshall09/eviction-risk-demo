# cosign-public-data-poc

This repository is a public-data underwriting risk proof of concept in Python.
The current version builds a county-level eviction-risk context feature that will later join to Cosign applicant scoring.

## Required input file

Place a raw CSV at:

`data/raw/eviction_county_month.csv`

Required columns (case-insensitive matching is supported):

- `county_fips`: county identifier (string or int). It will be normalized to a zero-padded 5-character string.
- `month`: month timestamp (`YYYY-MM-01` or `YYYY-MM`, or another pandas-parseable monthly format).
- `eviction_filing_rate`: float eviction filing rate for that county-month.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Train model and write artifacts:

```bash
python -m src.main --task train_eviction_model
```

Outputs:

- `data/processed/eviction_features.csv`
- `models/eviction_risk_model.joblib`
- `reports/model_metrics.json`

Score all counties for latest month:

```bash
python -m src.main --task score_latest
```

Output:

- `reports/county_risk_scores_latest.csv`

Score one county (example FIPS `39049`):

```bash
python -m src.main --task score_county --fips 39049
```

## Next steps

Next, we will add broader public-data ingestion and a stronger baseline model, then join this county risk signal into applicant-level underwriting workflows.
