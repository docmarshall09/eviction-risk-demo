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

## API (local)

Run the API server from your virtual environment:

```bash
source .venv/bin/activate
python -m src.main --task serve_api
```

Interactive docs:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

`/score` responses now include demo-friendly context fields:
- `features_used`: the four model inputs for that county-year.
- `risk_percentile_in_year`: percentile rank (0-100) among scoreable counties in that year.
- `as_of_year_available`, `available_years_min`, `available_years_max`: availability context for requested year/county.

## Environment variables

- `PORT`: API server port for `serve_api` (default `8000`).
- `API_CORS_ORIGINS`: Comma-separated allowlist for CORS origins. Leave unset for no CORS.

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

## Backtesting

Run leakage-safe yearly backtests using outcome-year holdouts:

```bash
python -m src.main --task backtest_eviction_lab_yearly
```

Generate the full backtest markdown summary (also refreshes backtest artifacts):

```bash
python -m src.main --task report_eviction_lab_backtest
```

This command runs two evaluations:
- Last outcome-year holdout (for example, 2018 when that is the latest year).
- Last two outcome-years holdout (for example, 2017 and 2018).

Artifacts written:
- `reports/eviction_lab_yearly_backtest_last_year.json`
- `reports/eviction_lab_yearly_backtest_last_2_years.json`
- `reports/eviction_lab_yearly_holdout_detail_last_year.csv`
- `reports/eviction_lab_yearly_holdout_detail_last_2_years.csv`

## Final retrain

After reviewing backtest quality, retrain the yearly model on all labeled rows:

```bash
python -m src.main --task train_eviction_lab_yearly_final
```

Then generate latest-year county scores:

```bash
python -m src.main --task score_eviction_lab_latest_year
```

## Next steps

Next, we will join these county-level context scores to applicant-level Cosign underwriting features and expand the modeling baseline.
