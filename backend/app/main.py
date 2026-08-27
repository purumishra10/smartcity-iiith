from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.database import Analysis, SessionLocal, init_db
from app.inference import QualityEngine
from app.vision.features import bgr_to_working, decode_image, extract_global_features
from app.vision.fusion import fuse_report
from app.vision.heatmaps import save_heatmap_overlays

log = logging.getLogger("clinic")
logging.basicConfig(level=logging.INFO)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
HEATMAP_KINDS = {"blur", "exposure", "noise", "defect"}

engine_ml = QualityEngine(settings.model_path, settings.scaler_path)

app = FastAPI(title="Civic Image Quality Clinic", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
    engine_ml.load()


def _analysis_dir(analysis_id: str) -> Path:
    path = Path(settings.storage_dir) / analysis_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_public(row: Analysis) -> dict:
    payload = json.loads(row.payload_json)
    payload["id"] = row.id
    payload["created_at"] = row.created_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    payload["heatmaps"] = {
        kind: f"/api/analyses/{row.id}/heatmaps/{kind}" for kind in HEATMAP_KINDS
    }
    payload["image_url"] = f"/api/analyses/{row.id}/image"
    payload["thumbnail_url"] = f"/api/analyses/{row.id}/image?thumb=1"
    return payload


@app.get("/health")
def health() -> dict:
    if not engine_ml.ready:
        raise HTTPException(status_code=503, detail=engine_ml.error or "model_not_loaded")
    return {"status": "ok", "model": "quality_mlp", "device": "cpu"}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    context: str | None = Form(default=None),
) -> dict:
    if not engine_ml.ready:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES and not (file.filename or "").lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    ):
        raise HTTPException(status_code=400, detail="unsupported_file_type")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty_file")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=400, detail="file_too_large")

    try:
        bgr = decode_image(data)
    except ValueError:
        raise HTTPException(status_code=400, detail="unreadable_image") from None

    analysis_id = str(uuid.uuid4())
    dest = _analysis_dir(analysis_id)
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    original_path = dest / f"original{suffix}"
    original_path.write_bytes(data)

    working = bgr_to_working(bgr, settings.max_analysis_dim)
    features, stats, maps = extract_global_features(working, grid_size=settings.grid_size)
    try:
        score, probs = engine_ml.predict(features)
    except Exception:
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference_failed") from None

    report = fuse_report(score, probs, stats)
    save_heatmap_overlays(maps, working.shape, dest / "heatmaps")

    thumb = cv2.resize(working, (160, 120), interpolation=cv2.INTER_AREA)
    thumb_path = dest / "thumb.jpg"
    cv2.imwrite(str(thumb_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    grid = {
        "rows": maps["rows"],
        "cols": maps["cols"],
        "tiles": maps["tiles"],
    }
    payload = {
        **report,
        "context": context,
        "grid": grid,
    }
    now = datetime.now(timezone.utc)
    db = SessionLocal()
    try:
        row = Analysis(
            id=analysis_id,
            created_at=now,
            quality_score=report["quality_score"],
            quality_label=report["quality_label"],
            context=context,
            original_path=str(original_path),
            thumbnail_path=str(thumb_path),
            payload_json=json.dumps(payload),
            file_size=len(data),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _to_public(row)
    except Exception:
        db.rollback()
        log.exception("persist failed")
        raise HTTPException(status_code=500, detail="persist_failed") from None
    finally:
        db.close()


@app.get("/api/analyses")
def list_analyses() -> list[dict]:
    db = SessionLocal()
    try:
        rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(100).all()
        out = []
        for row in rows:
            out.append(
                {
                    "id": row.id,
                    "created_at": row.created_at.replace(tzinfo=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "quality_score": row.quality_score,
                    "quality_label": row.quality_label,
                    "context": row.context,
                    "thumbnail_url": f"/api/analyses/{row.id}/image?thumb=1",
                    "image_url": f"/api/analyses/{row.id}/image",
                }
            )
        return out
    finally:
        db.close()


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    db = SessionLocal()
    try:
        row = db.get(Analysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not_found")
        return _to_public(row)
    finally:
        db.close()


@app.get("/api/analyses/{analysis_id}/image")
def get_image(analysis_id: str, thumb: int = 0):
    db = SessionLocal()
    try:
        row = db.get(Analysis, analysis_id)
        if row is None:
            raise HTTPException(status_code=404, detail="not_found")
        path = Path(row.thumbnail_path or row.original_path) if thumb else Path(row.original_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="file_missing")
        media = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        return FileResponse(path, media_type=media)
    finally:
        db.close()


@app.get("/api/analyses/{analysis_id}/heatmaps/{kind}")
def get_heatmap(analysis_id: str, kind: str):
    if kind not in HEATMAP_KINDS:
        raise HTTPException(status_code=400, detail="unknown_heatmap")
    path = Path(settings.storage_dir) / analysis_id / "heatmaps" / f"{kind}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="image/png")
