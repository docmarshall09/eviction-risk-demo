# Deploy to Fly.io

Run these commands from the repo root (`/Users/marshalldeese/Projects/cosign-public-data-poc`).

```bash
export APP=cosign-public-data-poc-demo
export REGION=iad

# 1) Login
fly auth login

# 2) Create Fly app config (no deploy yet)
fly launch --name "$APP" --region "$REGION" --ha=false --no-deploy --now

# 3) Create persistent volumes for runtime artifacts
fly volumes create "${APP}_data" --app "$APP" --region "$REGION" --size 3
fly volumes create "${APP}_models" --app "$APP" --region "$REGION" --size 3
fly volumes create "${APP}_reports" --app "$APP" --region "$REGION" --size 1
```

Append mounts to `fly.toml`:

```bash
cat >> fly.toml <<EOF

[[mounts]]
  source = "${APP}_data"
  destination = "/app/data"

[[mounts]]
  source = "${APP}_models"
  destination = "/app/models"

[[mounts]]
  source = "${APP}_reports"
  destination = "/app/reports"
EOF
```

Deploy, upload local artifacts, and verify:

```bash
# 4) Deploy container
fly deploy --app "$APP"

# 5) Upload required runtime artifacts via SFTP (local files -> mounted volumes)
fly ssh sftp put --app "$APP" data/processed/eviction_lab_yearly_features.csv /app/data/processed/eviction_lab_yearly_features.csv
fly ssh sftp put --app "$APP" models/eviction_lab_yearly_model.joblib /app/models/eviction_lab_yearly_model.joblib
fly ssh sftp put --app "$APP" models/eviction_lab_yearly_model_metadata.json /app/models/eviction_lab_yearly_model_metadata.json
fly ssh sftp put --app "$APP" reports/eviction_lab_yearly_model_metrics.json /app/reports/eviction_lab_yearly_model_metrics.json

# 6) Verify public URLs
curl -sS "https://${APP}.fly.dev/health"
curl -I "https://${APP}.fly.dev/demo/"
curl -I "https://${APP}.fly.dev/docs"
curl -sS -X POST "https://${APP}.fly.dev/score" \
  -H "Content-Type: application/json" \
  -d '{"county_fips":"39049","as_of_year":2015}'
curl -sS -X POST "https://${APP}.fly.dev/score" \
  -H "Content-Type: application/json" \
  -d '{"county_fips":"39049"}'
```

This deploy flow does not commit or bake proprietary CSV/model artifacts into the image.
