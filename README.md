# Civic Image Quality Clinic

Operator tool for civic stills (street, CCTV grab, incident photo). Upload an image; the clinic returns a quality score, a label (`ACCEPTABLE` / `DEGRADED` / `DEFECTIVE`), a plain-English diagnosis, per-issue confidence, and **clickable heatmaps** showing *where* blur, exposure, noise, and defects sit.

No external AI APIs. Inference runs locally on CPU.

## Layout

```text
backend/     FastAPI, SQLite, shared vision + MLP
frontend/     React (Vite) clinic UI
ml/           training, synthetic degradations, artifacts/
sample_images/  one example per quality condition
```

## How inference works

1. Decode and validate the upload.
2. Resize a working copy (max 512 px) and extract the same CV features globally and on a 16×16 tile grid (`backend/app/vision/features.py`).
3. Standardize the 12-D vector with `ml/artifacts/scaler.json` (train-set mean/std only).
4. `QualityMLP` (`model.pt`) predicts a 0–100 score and six issue probabilities.
5. Fusion rules assign the label, issue list, and diagnosis (`fusion.py`).
6. Tile maps are upsampled to PNG overlays. Global stats are the same formulas as the tiles so the map cannot invent a problem the score never saw.
7. SQLite stores the JSON; files live under `data/storage/<id>/`.

## Local setup (no Docker)

Python 3.11+ and Node 20+ recommended.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
# CPU torch if the default wheel pulls GPU:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Train (once) from the repo root, with the venv active:

```bash
python ml/train.py
python ml/generate_samples.py
```

This writes `ml/artifacts/model.pt`, `scaler.json`, `metrics.json`, and `sample_images/`.

API:

```bash
cd backend
set PYTHONPATH=.
uvicorn app.main:app --reload --port 8000
```

UI:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 (Vite proxies `/api` and `/health` to port 8000).

### Environment

Copy `.env.example`. Important variables:

| Variable | Meaning |
|----------|---------|
| `DATABASE_URL` | SQLite URL (default `data/clinic.db` at repo root) |
| `STORAGE_DIR` | originals + heatmaps |
| `MODEL_PATH` / `SCALER_PATH` | MLP artifacts |
| `MAX_UPLOAD_BYTES` | default 10 MB |
| `GRID_SIZE` | default 16 |
| `CORS_ORIGINS` | comma-separated origins |
| `VITE_API_URL` | leave empty when using the Vite proxy or nginx |

## Docker Compose

Requires trained artifacts in `ml/artifacts/` first (`python ml/train.py`).

```bash
docker compose up --build
```

- UI: http://localhost:8080
- API: http://localhost:8000
- Health: http://localhost:8000/health
- Database volume: `clinic_data` → `/data/clinic.db` and `/data/storage`

## API

`GET /health` — 200 if the model loaded, 503 otherwise.

`POST /api/analyze` — multipart field `file`, optional `context` (`street` | `camera` | `other`).

```bash
curl -F "file=@sample_images/01_blur.jpg" -F "context=street" http://localhost:8000/api/analyze
```

Example body:

```json
{
  "quality_score": 82,
  "quality_label": "ACCEPTABLE",
  "diagnosis": "...",
  "issues": [{"type": "noise", "severity": "low", "confidence": 0.71}],
  "statistics": {"sharpness": 0.01, "brightness": 0.4, "contrast": 0.2, "noise_estimate": 0.02, "saturation": 0.3},
  "heatmaps": {
    "blur": "/api/analyses/{id}/heatmaps/blur",
    "exposure": "/api/analyses/{id}/heatmaps/exposure",
    "noise": "/api/analyses/{id}/heatmaps/noise",
    "defect": "/api/analyses/{id}/heatmaps/defect"
  },
  "grid": {"rows": 16, "cols": 16}
}
```

`GET /api/analyses` — history.  
`GET /api/analyses/{id}` — full visit.  
`GET /api/analyses/{id}/image` — original (`?thumb=1` for thumbnail).  
`GET /api/analyses/{id}/heatmaps/{blur|exposure|noise|defect}` — overlay PNG.

Errors: 400 unreadable / wrong type / too large; 404 unknown id; 500 unexpected (no stack traces).

## Training recipe

See `EVALUATION.md`. Clean stills are procedural civic scenes; labels come from synthetic blur, exposure, noise, JPEG smash, and local defects. Holdout uses a different seed and stronger degradation ranges.

## Tests

```bash
cd backend
set PYTHONPATH=.
pytest -q
```
