# Web Demo

Static HTML/CSS/JS demo pages for the Eviction Lab yearly risk API.

## Run locally

Start the API from the repository root:

```bash
source .venv/bin/activate
python -m src.main --task serve_api
```

Then open:

- `http://127.0.0.1:8000/demo`

Useful routes:

- `http://127.0.0.1:8000/demo/` (Demo)
- `http://127.0.0.1:8000/demo/about.html` (About)
- `http://127.0.0.1:8000/demo/api.html` (API quickstart)
- `http://127.0.0.1:8000/docs` (Swagger UI)
