# Dr. Image

**Dr. Image** is an automated image quality assessment (IQA) and visual diagnostics platform engineered for civic stills (surveillance CCTV, traffic cameras, and field incident captures). It provides comprehensive image health diagnostics: a continuous 0–100 quality score, a tripartite clinical label (`ACCEPTABLE`, `DEGRADED`, `DEFECTIVE`), multi-head degradation probabilities, interpretable vital statistics, and interactive spatial heatmaps isolating blur, exposure defects, noise, and structural artifacts.

All inference runs locally on CPU with zero external API calls, combining classical vision feature extraction, a hybrid CNN+MLP quality estimator, and local YOLOv8 scene detection. Users can inspect heatmaps, preview/download PDF reports, and persist examination history across sessions.

## Layout

```text
backend/        FastAPI backend, SQLAlchemy ORM, SQLite/PostgreSQL, hybrid CNN+MLP engine
frontend/       React + Vite interactive diagnostic dashboard & report viewer
ml/             PyTorch training pipeline, degradation transforms, artifacts (model.pt, metrics)
sample_images/  Curated validation test stills (clean, blur, noise, exposure, defects)
```

## How inference works

1. **Intake & Preprocessing:** Uploaded image is decoded, validated, and normalized to a max dimension of 512 px while preserving aspect ratio.
2. **Feature Extraction:** Computes global statistical signals (MSCN natural scene statistics, FFT frequency energy, CLAHE residuals, chromatic cast, glare) and local 16×16 spatial grid statistics (`backend/app/vision/features.py`).
3. **Z-Score Normalization:** Transforms extracted features using precomputed training distribution parameters (`ml/artifacts/scaler.json`).
4. **Hybrid Neural Estimation:** `QualityHybrid` (`model.pt`) fuses a 128×128 RGB convolutional embedding with the MLP feature projection to predict the overall quality score and 6 degradation heads. A lightweight **YOLOv8n** pass detects civic entities (pedestrians, vehicles, signals) for context grounding.
5. **Multi-Modal Decision Fusion:** Rule-based heuristics and gating thresholds synthesize the final clinical label, primary diagnosis, and actionable recommendations (`fusion.py`).
6. **Heatmap Generation:** Grid-level spatial statistics are upsampled to generate pixel-aligned PNG heatmaps for blur, exposure, noise, and defects.
7. **Persistence:** Analysis records and metadata are persisted to PostgreSQL (Docker) or SQLite (local), associating session visits with guest or authenticated accounts.

## Quickstart with Docker (Recommended)

Docker Compose starts the containerized PostgreSQL database, FastAPI inference backend, and Nginx-backed React frontend out-of-the-box.

```bash
docker compose up --build
```

- **Frontend Dashboard:** http://localhost:8080
- **FastAPI Backend:** http://localhost:8000
- **Health Endpoint:** http://localhost:8000/health
- **Postgres Volume:** `clinic_pg` (stores user accounts and exam history)
- **Artifacts Volume:** `clinic_files` (mapped to `/data/storage`)

## Local setup (without Docker)

Prerequisites: **Python 3.11+** and **Node.js 20+**.

### 1. Backend Setup

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
# From backend/ with venv activated
python -m uvicorn app.main:app --reload --port 8000
```

*Note: Database defaults to SQLite at `data/clinic.db`. Storage files are written to `data/storage/`.*

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

*If your directory path contains spaces on Windows, run `node node_modules/vite/bin/vite.js` directly.*

Open **http://localhost:5173** (Vite automatically proxies `/api` and `/health` to port 8000).

### 3. ML Retraining (Optional)

Pretrained weights (`model.pt`, `scaler.json`, `yolov8n.pt`) are included in `ml/artifacts/`. To retrain:

```bash
python ml/train.py
python ml/generate_samples.py
```

## Environment Configuration

Configure environment variables via `.env` (refer to `.env.example`):

| Variable | Default / Format | Description |
|----------|------------------|-------------|
| `DATABASE_URL` | `sqlite:///.../data/clinic.db` | Database connection string (SQLite or PostgreSQL) |
| `STORAGE_DIR` | `.../data/storage` | Directory storing uploaded originals and heatmap PNGs |
| `MODEL_PATH` | `ml/artifacts/model.pt` | Path to PyTorch hybrid quality model weights |
| `SCALER_PATH` | `ml/artifacts/scaler.json` | Path to feature standard scaler parameters |
| `YOLO_MODEL_PATH` | `ml/artifacts/yolov8n.pt` | Path to YOLOv8n weights for scene parsing |
| `JWT_SECRET` | `dev-change-me-clinic-jwt` | Secret key used for signing session auth tokens |
| `MAX_UPLOAD_BYTES`| `10485760` (10 MB) | Maximum permitted upload payload size |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Allowed CORS origins for cross-domain requests |

## Authentication & Session Model

| Capability | Guest User | Registered User |
|------------|------------|-----------------|
| Run instant image analysis | Yes | Yes |
| Interactive heatmaps & PDF export | Yes | Yes |
| View past examination history | Session only | Persistent account history |
| Guest session migration | N/A | Automatically claimed on signup/login |

## REST API Reference

- `GET /health` — Returns status 200 with engine readiness, 503 if weights unavailable.
- `POST /api/analyze` — Multipart form upload (`file`, optional `context`: `street` | `camera` | `other`).
- `GET /api/analyses` — Retrieves paginated examination history for authenticated user.
- `GET /api/analyses/{id}` — Retrieves full structured JSON diagnostics for an exam.
- `GET /api/analyses/{id}/image` — Streams the stored original image (`?thumb=1` for thumbnail).
- `GET /api/analyses/{id}/heatmaps/{kind}` — Streams overlay PNG (`blur`, `exposure`, `noise`, `defect`).
- `GET /api/analyses/{id}/report.pdf` — Generates and downloads the multi-page clinical PDF report.
- `POST /api/auth/signup` & `POST /api/auth/login` — Authentication endpoints issuing HTTP-only JWT cookies.

Example cURL test:
```bash
curl -F "file=@sample_images/01_blur.jpg" -F "context=street" http://localhost:8000/api/analyze
```

## Testing & CI

```bash
cd backend
python -m pytest -q
```

Continuous Integration runs on GitHub Actions validating test suites and frontend production bundles.
