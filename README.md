# Eviction Risk Scoring Engine

County-level eviction risk model with transparent coefficients, backtesting, and a live scoring API.

**What it does:** Predicts whether a U.S. county will land in the top quartile for eviction filing rates next year, using historical Eviction Lab data and a logistic regression model with fully exposed coefficients.

**Why it exists:** A hands-on project to deepen my skills in Python-based ML/regression modeling for credit risk evaluation — and to continue building fluency working alongside state-of-the-art coding agents as development partners. The goal was to build something real: a working risk scoring tool grounded in public eviction data that demonstrates interpretable modeling, production discipline, and insurance-relevant thinking.

**Live demo:** [eviction-risk-demo.onrender.com](https://eviction-risk-demo.onrender.com)

---

## Architecture

- **Model:** Logistic regression (C=0.1) trained on county-year panels (2000–2017 features → 2001–2018 outcomes). Four features: `lag_1` (most recent filing rate), `lag_3_mean_obs` (3-year observed mean), `lag_5_mean_obs` (5-year observed mean), and `years_since_last_obs` (data staleness weight). Calibration guardrails prevent overconfident probability saturation.
- **Backend:** FastAPI with Pydantic schemas, serving real-time county scoring
- **Frontend:** Vanilla JS dashboard for interactive county lookup with inline model explainer
- **Infrastructure:** Docker build-time training → baked model artifact → Render deployment
- **Testing:** Smoke tests via FastAPI TestClient, temporal leakage assertions in training pipeline
- **Provenance:** Every model artifact includes git SHA, library versions, and training timestamp

## Key Design Decisions

**Logistic regression is intentional.** Not because it's simple — because it's explainable. In regulated underwriting, you need to produce a reason for every decline. Transparent coefficients make that trivial. Gradient-boosted models score higher on Kaggle; interpretable models survive regulatory scrutiny.

**Temporal leakage guards are enforced at runtime.** The training pipeline asserts that feature years are strictly less than outcome years before any model fitting. This isn't a test — it's a hard stop that prevents the single most dangerous failure mode in time-series ML.

**Backtesting uses held-out outcome years, not random splits.** Train/test splits respect the time boundary that exists in production: you never have next year's data when you score today's applicant.

**Model metadata is a first-class artifact.** The `/metadata` endpoint exposes training provenance (git SHA, sklearn version, training timestamp), feature list, regularization parameters, and performance summary. This supports model governance and audit without external tooling.

## What We Learned

Eviction filing rates are persistent year to year. In backtests, a naive baseline using only `lag_1` comes close to the full model — recent history carries most of the signal. The multi-lag feature set (`lag_3_mean_obs`, `lag_5_mean_obs`) and the `years_since_last_obs` staleness weight add incremental lift and handle the irregular county-year panel more robustly.

## Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | Service health check |
| `/metadata` | GET | Model version, features, provenance, metrics summary |
| `/score` | POST | Score a county by FIPS code and optional year |
| `/demo/` | GET | Interactive frontend |

## Local Development

```bash
# Clone and install
git clone https://github.com/docmarshall09/eviction-risk-demo.git
cd eviction-risk-demo
pip install -r requirements.txt

# Train model and run backtest
python -m src.main --task train_eviction_lab_yearly
python -m src.main --task backtest_eviction_lab_yearly

# Start API locally
python -m src.main --task serve_api

# Run tests
pytest tests/ -v
```

### Docker

```bash
docker build -t eviction-risk-engine .
docker run -p 8000:8000 eviction-risk-engine
```

The Docker build downloads training data, trains the model, and bakes the artifact into the image. No external model registry required.

## Project Structure

```
src/
├── api/                  # FastAPI app, Pydantic schemas
├── config.py             # Paths, constants
├── datasets/             # Data loading and validation
├── features/             # Feature engineering
├── main.py               # CLI orchestration
├── models/               # Training, evaluation, scoring
├── pipelines/            # Training dataset construction
├── reporting/            # Backtest summary reports
├── services/             # Scoring service layer
└── validation/           # Temporal leakage guards
tests/
├── test_smoke.py         # API endpoint smoke tests
└── test_leakage.py       # Leakage assertion unit tests
web/                      # Frontend (vanilla HTML/CSS/JS)
```

## Roadmap

The current model uses lag-1, lag-3, and lag-5 county-level filing rates with a recency weight for data gaps. Backtesting showed that lag-1 alone is a strong standalone predictor for the 2000–2018 panel, but the multi-lag approach handles irregular county-year coverage more robustly. Next steps focus on extending coverage and granularity:

* More recent data: The Eviction Lab dataset ends at 2018. Incorporate post-COVID filing data as it becomes available to test whether historical patterns hold.
* Monthly granularity: Move from annual to monthly panels to capture seasonal patterns and enable faster model refresh cycles.
* Spatial signals: Test whether neighboring-county filing rates carry predictive power (spatial autocorrelation / contagion effects).
* Non-geographic applicant signals: Begin exploring applicant-level risk features — income stability, rent-to-income ratio, payment history, thin-file indicators — that complement geographic base rates in a full underwriting model.

## Data

Training data: Princeton Eviction Lab county-level proprietary dataset (2000–2018). Downloaded at Docker build time.

## License

MIT
