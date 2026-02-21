# cosign-public-data-poc

This repository is a public-data underwriting risk proof of concept in Python.
It includes a monthly county pipeline and a parallel yearly Eviction Lab pipeline for county-level context risk scoring.

## Required input files

Monthly pipeline input:
- `data/raw/eviction_county_month.csv`
- Required columns (case-insensitive): `county_fips`, `month`, `eviction_filing_rate`

Yearly Eviction Lab pipeline input:
- `data/raw/county_proprietary_valid_2000_2018.csv`
- Expected columns include: `cofips`, `county`, `state`, `year`, `filings`, `filing_rate`
- The yearly pipeline uses observed/validated annual rates and treats the data as an irregular panel (no gap-filling for missing years).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Monthly pipeline tasks

Train monthly model and write artifacts:

```bash
python -m src.main --task train_eviction_model
```

Score all counties for latest month:

```bash
python -m src.main --task score_latest
```

Score one county for latest month:

```bash
python -m src.main --task score_county --fips 39049
```

## Yearly Eviction Lab tasks

Train yearly model and write artifacts:

```bash
python -m src.main --task train_eviction_lab_yearly
```

Outputs:
- `data/processed/eviction_lab_yearly_features.csv`
- `models/eviction_lab_yearly_model.joblib`
- `reports/eviction_lab_yearly_model_metrics.json`

Score all counties for latest available year:

```bash
python -m src.main --task score_eviction_lab_latest_year
```

Output:
- `reports/eviction_lab_county_risk_scores_latest_year.csv`

Score one county for latest available year:

```bash
python -m src.main --task score_eviction_lab_county --fips 39049
```

## Next steps

Next, we will join these county-level context scores to applicant-level Cosign underwriting features and expand the modeling baseline.
