# Dr. Image

Operator tool for civic stills (street, CCTV grab, incident photo). Upload an image; Dr. Image returns a quality score, a label (`ACCEPTABLE` / `DEGRADED` / `DEFECTIVE`), a plain-English diagnosis, **why each vital scored as it did**, per-issue confidence, and **clickable heatmaps** showing where blur, exposure, noise, and defects sit. You can download a **PDF report** with the original still and all four overlays.

No external AI APIs. Inference runs locally on CPU.

**Accounts:** exams are free without logging in. **Past exams** (saved history) requires sign-up. Guest exams from the current browser session are attached to your account when you register or log in.

## Layout

```text
backend/     FastAPI, SQLAlchemy, Postgres or SQLite, hybrid CNN+MLP
frontend/     React (Vite) clinic UI
ml/           train, synthetic degrade, optional public stills, artifacts/
sample_images/  one example per quality condition
```

## How inference works

1. Decode and validate the upload.
2. Resize a working copy (max 512 px) and extract CV features globally and on a 16×16 tile grid (`backend/app/vision/features.py`). Extra full-image signals (FFT, MSCN, CLAHE residual, colour cast, glare) are concatenated.
3. Standardize the feature vector with `ml/artifacts/scaler.json` (train-set mean/std only).
4. `QualityHybrid` (`model.pt`) concatenates a tiny CNN embedding (128×128 RGB) with the CV MLP embedding, then predicts a 0–100 score and six issue probabilities. A local **YOLOv8n** pass (COCO) lists people, cars, buses, bikes, and traffic lights for the left-side scene text.
5. Fusion rules assign the label, issue list, diagnosis, and structured explanations (`fusion.py`).
6. Tile maps are upsampled to PNG overlays. Global stats use the same formulas as the tiles so the map cannot invent a problem the score never saw.
7. The visit is stored (Postgres in Docker, SQLite locally). Files live under `STORAGE_DIR/<id>/`. Guests are keyed by an httpOnly session cookie; logged-in users own the row.

## Local setup (no Docker)

Python 3.11+ and Node 20+ recommended.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix: source .venv/bin/activate
pip install -r requirements.txt
```

Train (once) from the repo root, with the venv active:

```bash
python ml/train.py
python ml/generate_samples.py
```

This writes `ml/artifacts/model.pt`, `scaler.json`, `metrics.json`, and `sample_images/`. Training tries a small Kodak download into `ml/data/public/`; if the network is blocked it falls back to procedural streets only.

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

If the folder path contains spaces (`Hackathons & Competitions`), `npm run dev` may fail on Windows. From `frontend/` run:

```bash
node node_modules/vite/bin/vite.js
```

Open http://localhost:5173 (Vite proxies `/api` and `/health` to port 8000 so auth cookies stay on the UI origin).

Local database defaults to **SQLite** at `smartcity-iiith/data/clinic.db`. Do not delete `data/` if you want history to survive restarts. Uploads: `data/storage/`.

### Environment

Copy `.env.example`. Important variables:

| Variable | Meaning |
|----------|---------|
| `DATABASE_URL` | SQLite file URL, or `postgresql+psycopg2://…` |
| `STORAGE_DIR` | originals + heatmaps |
| `MODEL_PATH` / `SCALER_PATH` | hybrid artifacts |
| `YOLO_MODEL_PATH` | YOLOv8n weights for “what's in the frame” (auto-download on first run) |
| `JWT_SECRET` | signs login cookies |
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
- Postgres volume `clinic_pg` (history)
- File volume `clinic_files` → `/data/storage`

Set `JWT_SECRET` in the environment for anything beyond a laptop demo.

Guest rows that are never claimed by an account remain on disk for the guest cookie lifetime (7 days on the cookie). Orphan cleanup is not automated; operators can wipe volumes.

## Auth behaviour

| Action | Guest | Logged in |
|--------|-------|-----------|
| Run exam | yes | yes |
| Open that exam / PDF (same browser) | yes | yes |
| Past exams list | empty + CTA | own rows only |
| After sign-up / log-in | session exams become saved | merged |

## API

`GET /health` — 200 if the model loaded, 503 otherwise.

`POST /api/auth/signup` / `POST /api/auth/login` — JSON `{ "email", "password" }` (password ≥ 8). Sets httpOnly `clinic_token`.

`POST /api/auth/logout`

`GET /api/me`

`POST /api/analyze` — multipart field `file`, optional `context` (`street` | `camera` | `other`). No login required.

```bash
curl -F "file=@sample_images/01_blur.jpg" -F "context=street" http://localhost:8000/api/analyze
```

The JSON includes `diagnosis`, `vital_explanations`, `issue_explanations`, `heatmaps`, `report_url`, and `saved_to_history`.

`GET /api/analyses` — history for the **logged-in user** only (empty list if guest).

`GET /api/analyses/{id}` — full visit if you own it or hold the guest session.

`GET /api/analyses/{id}/image` — original (`?thumb=1` for thumbnail).

`GET /api/analyses/{id}/heatmaps/{blur\|exposure\|noise\|defect}` — overlay PNG.

`GET /api/analyses/{id}/report.pdf` — detailed PDF (original + four composited heatmaps + write-ups).

Errors: 400 unreadable / wrong type / too large; 401 bad login; 404 unknown or not yours; 409 email taken; 500 unexpected (no stack traces).

## Training recipe

See `EVALUATION.md`.

## Tests

```bash
cd backend
set PYTHONPATH=.
pytest -q
```

CI (GitHub Actions) runs pytest and `npm run build`.
